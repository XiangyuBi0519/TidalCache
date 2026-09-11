"""
TidalCache Offload Manager

Manages the full lifecycle of sparse-attention KV Cache offloading:
  1. Host hugepage allocation + NPU MMU registration (HostKVPool)
  2. Per-layer Device Selection Cache buffers (SelectionCache)
  3. GatherSelectionKvCache operator calls

Supports both DeepSeek-V3 (DSA) and DeepSeek-V4 (CSA) sparse attention.
HCA layers (dense attention with heavy compression) are excluded — they
don't use Lightning Indexer / top-k selection.

Usage in vllm-ascend:
  - Instantiate TidalCacheManager at model init time
  - Call alloc_layer() for each sparse attention layer (DSA/CSA only)
  - Call gather() in _forward_decode() after Lightning Indexer produces topk_idxs
"""

import os
import mmap
import logging
from dataclasses import dataclass

import torch

from tidalcache import HUGEPAGE_PATH

logger = logging.getLogger("tidalcache")

HUGEPAGE_SIZE = 2 * 1024 * 1024  # 2MB


@dataclass
class LayerOffloadState:
    """Per-layer offload state."""
    # Host Full KV (hugepage, CPU tensor + NPU view)
    host_kv_cache: torch.Tensor       # CPU tensor on hugepage
    host_k_rope: torch.Tensor         # CPU tensor on hugepage
    npu_kv_cache: torch.Tensor        # NPU view of host_kv_cache
    npu_k_rope: torch.Tensor          # NPU view of host_k_rope

    # Device Selection Cache
    sel_kv_cache: torch.Tensor        # [sel_blocks, block_size, kv_dim]
    sel_k_rope: torch.Tensor          # [sel_blocks, block_size, rope_dim]
    sel_block_table: torch.Tensor     # [max_batch, topk]
    sel_block_status: torch.Tensor    # [max_batch, 1, 1, topk+1]

    # Hugepage cleanup handles
    mmap_kv: mmap.mmap
    mmap_rope: mmap.mmap
    fd_kv: int
    fd_rope: int
    hugepage_path_kv: str
    hugepage_path_rope: str


class TidalCacheManager:
    """
    Central manager for TidalCache KV offloading.

    One instance per worker, manages all DSA layers.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        kv_dim: int,
        rope_dim: int,
        index_topk: int,
        max_batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rope_dtype: torch.dtype | None = None,
        compress_block_size: int = 64,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.compress_block_size = compress_block_size
        self.kv_dim = kv_dim
        self.rope_dim = rope_dim
        self.index_topk = index_topk
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.rope_dtype = rope_dtype if rope_dtype is not None else dtype
        self.device = device

        self.layers: dict[str, LayerOffloadState] = {}
        self._zero_copy_npu = None

        logger.info(
            "TidalCache init: blocks=%d, block_size=%d, compress_block_size=%d, "
            "kv_dim=%d, rope_dim=%d, topk=%d, max_batch=%d, kv_dtype=%s, rope_dtype=%s",
            num_blocks, block_size, compress_block_size, kv_dim, rope_dim,
            index_topk, max_batch_size, dtype, self.rope_dtype,
        )

    def _get_zero_copy(self):
        if self._zero_copy_npu is None:
            import zero_copy_npu
            self._zero_copy_npu = zero_copy_npu
        return self._zero_copy_npu

    def _get_gather_wrapper(self):
        if not hasattr(self, '_gather_wrapper'):
            import ctypes
            cann = os.environ.get(
                "ASCEND_TOOLKIT_HOME",
                os.environ.get("ASCEND_HOME_PATH",
                               "/usr/local/Ascend/cann-9.0.1"))
            lib_path = os.path.join(
                cann, "opp/vendors/customize/op_api/lib/libcust_opapi.so")
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            import gather_wrapper
            self._gather_wrapper = gather_wrapper
        return self._gather_wrapper

    # ── Host Hugepage Allocation ──

    def _alloc_hugepage_tensor(
        self, shape: list[int], dtype: torch.dtype, name: str
    ) -> tuple[torch.Tensor, mmap.mmap | None, int, str]:
        """Allocate a tensor on hugepage via mmap, fallback to pinned memory.

        Returns (cpu_tensor, mmap_obj_or_None, fd_or_-1, path_or_empty).
        """
        numel = 1
        for s in shape:
            numel *= s
        elem_size = torch.tensor([], dtype=dtype).element_size()
        data_bytes = numel * elem_size
        aligned_size = ((data_bytes + HUGEPAGE_SIZE - 1)
                        // HUGEPAGE_SIZE) * HUGEPAGE_SIZE

        try:
            path = os.path.join(HUGEPAGE_PATH, f"tidalcache_{name}_{id(self)}")
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, aligned_size)
            mmap_obj = mmap.mmap(
                fd, aligned_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            tensor = torch.frombuffer(
                mmap_obj, dtype=dtype, count=numel
            ).view(shape)
            tensor.zero_()
            logger.info(
                "Hugepage alloc: %s shape=%s bytes=%d path=%s",
                name, shape, data_bytes, path,
            )
            return tensor, mmap_obj, fd, path
        except OSError as e:
            logger.warning(
                "Hugepage alloc failed for %s (%s), falling back to pinned memory",
                name, e,
            )
            tensor = torch.zeros(shape, dtype=dtype, pin_memory=True)
            logger.info(
                "Pinned memory alloc: %s shape=%s bytes=%d",
                name, shape, data_bytes,
            )
            return tensor, None, -1, ""

    def _register_npu(self, host_tensor: torch.Tensor) -> torch.Tensor:
        """Register host tensor to NPU MMU, return NPU view tensor."""
        zcn = self._get_zero_copy()
        device_id = self.device.index if self.device.index is not None else 0
        _, npu_tensor = zcn.register_hugepage_as_npu_tensor(
            host_tensor, device_id)
        return npu_tensor

    # ── Per-Layer Allocation ──

    def alloc_layer(self, layer_name: str,
                    local_topk: int | None = None) -> LayerOffloadState:
        """Allocate Host + Device buffers for one sparse attention layer.

        Call this during _allocate_kv_cache_tensors() for each DSA/CSA layer.
        HCA layers should NOT call this — they use dense attention.

        Args:
            local_topk: Actual per-rank topk (= index_topk // cp_size in CP
                        mode). Defaults to self.index_topk for non-CP.
        """
        if layer_name in self.layers:
            return self.layers[layer_name]

        topk = local_topk if local_topk is not None else self.index_topk
        safe_name = layer_name.replace(".", "_")

        # Host Full KV Cache (hugepage)
        host_kv, mmap_kv, fd_kv, path_kv = self._alloc_hugepage_tensor(
            [self.num_blocks, self.block_size, self.kv_dim],
            self.dtype,
            f"{safe_name}_kv",
        )
        host_rope, mmap_rope, fd_rope, path_rope = self._alloc_hugepage_tensor(
            [self.num_blocks, self.block_size, self.rope_dim],
            self.rope_dtype,
            f"{safe_name}_rope",
        )

        # Register to NPU MMU
        npu_kv = self._register_npu(host_kv)
        npu_rope = self._register_npu(host_rope)

        logger.info(
            "Layer %s: host_kv ptr=0x%x → npu ptr=0x%x (local_topk=%d)",
            layer_name, host_kv.data_ptr(), npu_kv.data_ptr(), topk,
        )

        # Device Selection Cache — uses compress_block_size (not KV block_size)
        sel_blocks = self.max_batch_size * topk
        sel_kv = torch.zeros(
            sel_blocks, self.compress_block_size, self.kv_dim,
            dtype=self.dtype, device=self.device,
        )
        sel_rope = torch.zeros(
            sel_blocks, self.compress_block_size, self.rope_dim,
            dtype=self.rope_dtype, device=self.device,
        )
        sel_block_table = torch.arange(
            sel_blocks, dtype=torch.int32, device=self.device,
        ).view(self.max_batch_size, topk)
        sel_block_status = torch.full(
            (self.max_batch_size, 1, 1, topk + 1),
            -1, dtype=torch.int32, device=self.device,
        )

        state = LayerOffloadState(
            host_kv_cache=host_kv,
            host_k_rope=host_rope,
            npu_kv_cache=npu_kv,
            npu_k_rope=npu_rope,
            sel_kv_cache=sel_kv,
            sel_k_rope=sel_rope,
            sel_block_table=sel_block_table,
            sel_block_status=sel_block_status,
            mmap_kv=mmap_kv,
            mmap_rope=mmap_rope,
            fd_kv=fd_kv,
            fd_rope=fd_rope,
            hugepage_path_kv=path_kv,
            hugepage_path_rope=path_rope,
        )
        self.layers[layer_name] = state
        return state

    # ── GatherSelectionKvCache ──

    def gather(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        full_block_table: torch.Tensor,
        full_actual_seq: torch.Tensor,
        full_q_actual_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute GatherSelectionKvCache for one layer.

        Args:
            layer_name: layer identifier (e.g. "model.layers.3.self_attn")
            topk_indices: [B, 1, 1, topk] INT32 from Lightning Indexer
            full_block_table: [B, max_blocks] INT32 compressor block table
            full_actual_seq: [B] INT32 actual sequence lengths
            full_q_actual_seq: [B] INT32 query lengths (1 for decode)

        Returns:
            (sel_kv_cache, sel_k_rope, sel_actual_seq)
            - sel_kv_cache: Device tensor with gathered KV data
            - sel_k_rope: Device tensor with gathered RoPE data
            - sel_actual_seq: [B] actual sequence lengths for selection
        """
        state = self.layers[layer_name]
        gw = self._get_gather_wrapper()

        # Use full_block_table B as canonical batch size — topk_indices may
        # be padded (e.g. during graph capture warmup).
        batch_size = full_block_table.shape[0]
        topk_indices = topk_indices[:batch_size]
        full_actual_seq = full_actual_seq[:batch_size]
        full_q_actual_seq = full_q_actual_seq[:batch_size]
        sel_block_table = state.sel_block_table[:batch_size]
        sel_block_status = state.sel_block_status[:batch_size]

        sel_actual_seq = gw.npu_gather_selection_kv_cache(
            state.sel_k_rope,
            state.sel_kv_cache,
            sel_block_table,
            sel_block_status,
            topk_indices,
            state.npu_k_rope,
            state.npu_kv_cache,
            full_block_table,
            full_actual_seq,
            full_q_actual_seq,
            self.compress_block_size,
        )

        return state.sel_kv_cache, state.sel_k_rope, sel_actual_seq

    # ── Batch Lifecycle ──

    def reset_requests(self, layer_name: str, batch_indices: torch.Tensor):
        """Reset block_status for finished/new requests.

        Call when requests enter or leave a batch slot, so GatherSelectionKvCache
        knows those slots have no valid cached blocks.
        """
        state = self.layers[layer_name]
        for idx in batch_indices.tolist():
            state.sel_block_status[idx].fill_(-1)

    def reset_all(self, layer_name: str):
        """Reset all batch slots for a layer."""
        state = self.layers[layer_name]
        state.sel_block_status.fill_(-1)

    # ── Cleanup ──

    def cleanup(self):
        """Release all hugepage allocations and NPU registrations."""
        zcn = self._get_zero_copy()
        for name, state in self.layers.items():
            try:
                zcn.unregister_host(state.host_kv_cache)
                zcn.unregister_host(state.host_k_rope)
            except Exception as e:
                logger.warning("Unregister failed for %s: %s", name, e)
            try:
                state.mmap_kv.close()
                state.mmap_rope.close()
                os.close(state.fd_kv)
                os.close(state.fd_rope)
                os.unlink(state.hugepage_path_kv)
                os.unlink(state.hugepage_path_rope)
            except Exception as e:
                logger.warning("Cleanup failed for %s: %s", name, e)
        self.layers.clear()
        logger.info("TidalCache cleanup complete")

    def __del__(self):
        if self.layers:
            self.cleanup()
