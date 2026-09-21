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
import atexit
import signal
import weakref
import glob
import logging
from dataclasses import dataclass

import torch

from tidalcache import HUGEPAGE_PATH

logger = logging.getLogger("tidalcache")

HUGEPAGE_SIZE = 2 * 1024 * 1024  # 2MB
TOPK_SPLIT_NUM = 32  # CANN operator requires block_size=1 when topk>32; split to stay on scalar path

# ── Hugepage leak prevention ─────────────────────────────────────────────
# Track every live manager so we can release its files on process exit.
_LIVE_MANAGERS: "weakref.WeakSet" = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False


def _cleanup_all_managers(*_args, **_kwargs):
    """atexit / signal handler: unlink hugepage files created by any live manager."""
    for mgr in list(_LIVE_MANAGERS):
        try:
            mgr.cleanup()
        except Exception as e:  # pragma: no cover — best-effort
            logger.warning("cleanup_all: manager cleanup failed: %s", e)


def _install_signal_handlers():
    """Install SIGTERM/SIGINT handlers ONCE per process."""
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            prev = signal.getsignal(sig)

            def _make_handler(_prev, _sig):
                def _handler(signum, frame):
                    _cleanup_all_managers()
                    # Chain to previous handler if it was callable
                    if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                        try:
                            _prev(signum, frame)
                            return
                        except SystemExit:
                            raise
                        except Exception:
                            pass
                    # Restore default and re-raise so the process exits cleanly
                    signal.signal(_sig, signal.SIG_DFL)
                    os.kill(os.getpid(), _sig)
                return _handler

            signal.signal(sig, _make_handler(prev, sig))
        except (ValueError, OSError):
            # Signals not settable in this thread (e.g. non-main). Skip.
            pass
    _SIGNAL_HANDLERS_INSTALLED = True


def _sweep_stale_hugepage_files():
    """Best-effort: remove `tidalcache_*` / `group_*` files from `/dev/hugepages/`
    that aren't held open by any live process.

    Enabled by env `TIDALCACHE_SWEEP_STALE=1`. Safe when this process is the
    only TidalCache instance on the host (typical single-service deployment).
    """
    if os.environ.get("TIDALCACHE_SWEEP_STALE", "0") != "1":
        return
    patterns = [
        os.path.join(HUGEPAGE_PATH, "tidalcache_*"),
        os.path.join(HUGEPAGE_PATH, "group_*"),
    ]
    swept = 0
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.unlink(path)
                swept += 1
            except OSError:
                pass  # file busy or gone — ok
    if swept:
        logger.info("[cleanup] swept %d stale hugepage files from %s", swept, HUGEPAGE_PATH)


# Register global cleanup once at module import.
atexit.register(_cleanup_all_managers)


@dataclass
class LayerOffloadState:
    """Per-layer offload state.

    Host side (host_*, npu_*) is allocated eagerly (e.g. at first scatter,
    when the manager doesn't know the actual per-rank topk).

    Device Selection side (sel_*) is allocated lazily at first gather with
    the correct local_topk (index_topk // cp_size in CP mode). Fields default
    to None until upgraded.
    """
    # Host Full KV (hugepage, CPU tensor + NPU view)
    host_kv_cache: torch.Tensor       # CPU tensor on hugepage
    host_k_rope: torch.Tensor         # CPU tensor on hugepage
    npu_kv_cache: torch.Tensor        # NPU view of host_kv_cache
    npu_k_rope: torch.Tensor          # NPU view of host_k_rope

    # Hugepage cleanup handles
    mmap_kv: mmap.mmap
    mmap_rope: mmap.mmap
    fd_kv: int
    fd_rope: int
    hugepage_path_kv: str
    hugepage_path_rope: str

    # Device Selection Cache — filled by _alloc_sel_side once local_topk known
    sel_kv_cache: "torch.Tensor | None" = None    # [sel_blocks, block_size, kv_dim]
    sel_k_rope: "torch.Tensor | None" = None      # [sel_blocks, block_size, rope_dim]
    sel_block_table: "torch.Tensor | None" = None # [max_batch, topk]
    sel_block_status: "torch.Tensor | None" = None


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
        self._alloc_count = 0
        # Track every hugepage file we create; cleanup() unlinks these.
        self._created_hugepage_paths: set[str] = set()

        # Register for atexit / signal cleanup, sweep stale files (opt-in).
        _sweep_stale_hugepage_files()
        _install_signal_handlers()
        _LIVE_MANAGERS.add(self)

        # B3 v2: Unified Host allocation — vllm's Host-backed compress_kv_cache
        # is reused as TidalCache's per-layer host_kv_cache. When this map has
        # an entry for a layer, `ensure_host_allocated` skips its own hugepage
        # alloc and points state.npu_kv_cache directly at the vllm view.
        # Populated by MR_PATCH2 after kv_caches finalization.
        self._layer_compress_view: dict[str, torch.Tensor] = {}

        # Read TAG for HBM logging (single-line HBM summary uses this)
        self.tag = os.environ.get("TIDALCACHE_TAG", "ON")

        logger.info(
            "TidalCache init: blocks=%d, block_size=%d, compress_block_size=%d, "
            "kv_dim=%d, rope_dim=%d, topk=%d, max_batch=%d, kv_dtype=%s, rope_dtype=%s",
            num_blocks, block_size, compress_block_size, kv_dim, rope_dim,
            index_topk, max_batch_size, dtype, self.rope_dtype,
        )

    def _log_hbm_rank0(self, stage: str):
        """Emit one HBM summary line. rank 0 only."""
        try:
            import torch_npu  # noqa
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return
            free, total = torch.npu.mem_get_info()
            GB = 1024 ** 3
            logger.info(
                "[HBM] %s [MODE=%s] | used=%.2f/%.2fGB | free=%.2fGB | tc_layers=%d",
                stage, self.tag,
                (total - free) / GB, total / GB, free / GB,
                self._alloc_count,
            )
        except Exception as e:
            logger.warning("HBM log failed: %s", e)

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
            self._created_hugepage_paths.add(path)
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

    def alloc_group_host_tensor(
        self, size_bytes: int, group_name: str = "compress",
    ) -> torch.Tensor:
        """Allocate a byte-buffer on Host hugepage and register as NPU MMU view.

        Returns an NPU-viewable tensor of shape (size_bytes,) with dtype int8.
        Intended to replace vllm's Device raw_tensor allocation for one
        kv_cache_group (e.g. the DSA compress group). vllm downstream will
        reinterpret/reshape via .view() as usual.

        This is the Phase B3 v2 core primitive: keep the tensor "on Device"
        from vllm's perspective (data_ptr in NPU address space), but the
        actual storage is on Host DDR (hugepage) so we save HBM.
        """
        safe = group_name.replace("/", "_").replace(".", "_")
        # Allocate as int8 to match vllm's byte-buffer allocation contract.
        host_tensor, mmap_obj, fd, path = self._alloc_hugepage_tensor(
            [size_bytes], torch.int8, f"group_{safe}_{id(self)}",
        )
        npu_tensor = self._register_npu(host_tensor)
        # Track for cleanup — reuse LayerOffloadState-style storage
        if not hasattr(self, "_group_allocations"):
            self._group_allocations = []
        self._group_allocations.append({
            "name": group_name,
            "size_bytes": size_bytes,
            "host": host_tensor,
            "npu": npu_tensor,
            "mmap": mmap_obj,
            "fd": fd,
            "path": path,
        })
        logger.info(
            "[GROUP-ALLOC] %s: %.2f MB → Host hugepage + NPU MMU view "
            "(host_ptr=0x%x, npu_ptr=0x%x)",
            group_name, size_bytes / 1024**2,
            host_tensor.data_ptr(), npu_tensor.data_ptr(),
        )
        return npu_tensor

    def _register_npu(self, host_tensor: torch.Tensor) -> torch.Tensor:
        """Register host tensor to NPU MMU, return NPU view tensor."""
        zcn = self._get_zero_copy()
        device_id = self.device.index if self.device.index is not None else 0
        _, npu_tensor = zcn.register_hugepage_as_npu_tensor(
            host_tensor, device_id)
        return npu_tensor

    # ── Per-Layer Allocation ──

    def ensure_host_allocated(self, layer_name: str) -> LayerOffloadState:
        """Allocate ONLY the Host side (hugepage + NPU MMU registration).

        Used by scatter (PATCH3) which needs the NPU view of Host to redirect
        writes there. sel_* fields stay None until first gather (alloc_layer)
        provides the correct local_topk — avoids over-allocating sel_kv by
        cp_size× in CP mode.

        B3 v2: if MR_PATCH2 pre-registered a compress_view for this layer
        (via `_layer_compress_view`), reuse it as the Host-backed source
        instead of allocating a fresh hugepage. This avoids doubling Host
        memory (one for vllm compress + one for TidalCache).
        """
        if layer_name in self.layers:
            return self.layers[layer_name]

        # B3 v2 fast path: reuse vllm's Host-backed compress view
        _shared_view = self._layer_compress_view.get(layer_name)
        if _shared_view is not None:
            # Small placeholder rope (fresh alloc for now — rope isn't shared
            # with vllm's compress which is KV-only). Kept minimal to avoid
            # HBM/Host pressure; can zero-fill since sparse attention rope is
            # not fed to compress path.
            safe_name = layer_name.replace(".", "_")
            host_num_blocks = self.num_blocks * (
                self.block_size // self.compress_block_size)
            host_rope, mmap_rope, fd_rope, path_rope = self._alloc_hugepage_tensor(
                [host_num_blocks, self.compress_block_size, self.rope_dim],
                self.rope_dtype,
                f"{safe_name}_rope",
            )
            npu_rope = self._register_npu(host_rope)
            # host_kv_cache and npu_kv_cache both point at vllm's compress view.
            # It's already Host-backed (via MR_PATCH3) AND NPU-viewable.
            logger.info(
                "Layer %s: reusing vllm's compress view as host_kv_cache "
                "(shape=%s, dtype=%s) — no fresh alloc",
                layer_name, tuple(_shared_view.shape), _shared_view.dtype,
            )
            state = LayerOffloadState(
                host_kv_cache=_shared_view,
                host_k_rope=host_rope,
                npu_kv_cache=_shared_view,  # same tensor — already NPU addressable
                npu_k_rope=npu_rope,
                mmap_kv=None,               # storage owned by vllm's group tensor
                mmap_rope=mmap_rope,
                fd_kv=-1,
                fd_rope=fd_rope,
                hugepage_path_kv="",
                hugepage_path_rope=path_rope,
            )
            state.sel_block_status_list = None
            state.local_topk = None
            self.layers[layer_name] = state
            return state

        safe_name = layer_name.replace(".", "_")

        # Host Full KV Cache (hugepage) — use compress_block_size so
        # f_blk_size == s_blk_size as required by the CANN operator.
        host_num_blocks = self.num_blocks * (
            self.block_size // self.compress_block_size)
        host_kv, mmap_kv, fd_kv, path_kv = self._alloc_hugepage_tensor(
            [host_num_blocks, self.compress_block_size, self.kv_dim],
            self.dtype,
            f"{safe_name}_kv",
        )
        host_rope, mmap_rope, fd_rope, path_rope = self._alloc_hugepage_tensor(
            [host_num_blocks, self.compress_block_size, self.rope_dim],
            self.rope_dtype,
            f"{safe_name}_rope",
        )

        # Register to NPU MMU
        npu_kv = self._register_npu(host_kv)
        npu_rope = self._register_npu(host_rope)

        logger.info(
            "Layer %s: host allocated (host_kv=0x%x → npu=0x%x); sel deferred",
            layer_name, host_kv.data_ptr(), npu_kv.data_ptr(),
        )

        state = LayerOffloadState(
            host_kv_cache=host_kv,
            host_k_rope=host_rope,
            npu_kv_cache=npu_kv,
            npu_k_rope=npu_rope,
            mmap_kv=mmap_kv,
            mmap_rope=mmap_rope,
            fd_kv=fd_kv,
            fd_rope=fd_rope,
            hugepage_path_kv=path_kv,
            hugepage_path_rope=path_rope,
        )
        state.sel_block_status_list = None
        state.local_topk = None
        self.layers[layer_name] = state
        return state

    def set_layer_compress_view(self, layer_name: str, view: torch.Tensor):
        """Register vllm's Host-backed compress tensor for this layer.

        Called by MR_PATCH2 after kv_caches is finalized. Downstream
        ensure_host_allocated will reuse this view instead of allocating a
        fresh hugepage — achieves the "single Host allocation" invariant
        for Phase B3 v2.
        """
        self._layer_compress_view[layer_name] = view

    def _alloc_sel_side(self, state: LayerOffloadState, local_topk: int):
        """Allocate Device Selection Cache side with correct local_topk.

        Allocates mini_compress_kv as a [mini_num_blocks, block_size=128, kv_dim]
        tensor (matches attn_op's expected PA_ND layout). sel_kv is a VIEW of
        the same storage in the [max_batch*topk, compress_block_size=64, kv_dim]
        layout that CANN gather wants (f_blk_size == s_blk_size).

        Layout mapping:
            gpb = block_size / compress_block_size  (usually 2)
            sel_kv[b*topk + k] ↔ mini_compress_kv[b*(topk/gpb) + k/gpb][k%gpb*64:(k%gpb+1)*64]
        This alias means gather writes → attention reads. No copy-back needed.
        """
        topk = local_topk
        gpb = self.block_size // self.compress_block_size
        assert topk % gpb == 0, (
            f"topk ({topk}) must be divisible by groups_per_block ({gpb})"
        )
        sel_blocks = self.max_batch_size * topk         # 64-group units
        mini_num_blocks = sel_blocks // gpb              # 128-block units

        # Allocate flat storage and create two views:
        #   - mini_compress_kv 4D PA_ND {Bn, Bs, N=1, D} — attn_op reads this
        #   - sel_kv           3D {sel_blocks, compress_block_size, D} — gather writes here
        # Views share storage so gather → attn is zero-copy.
        # MLA's num_heads for KV is 1 (single latent head), matching vllm's
        # compress_kv_cache layout so SparseAttnSharedkv layout parser reads
        # N2=1 (matches ori_kv's N2=1).
        _kv_numel = mini_num_blocks * self.block_size * self.kv_dim
        _rope_numel = mini_num_blocks * self.block_size * self.rope_dim
        _kv_storage = torch.zeros(_kv_numel, dtype=self.dtype, device=self.device)
        _rope_storage = torch.zeros(_rope_numel, dtype=self.rope_dtype, device=self.device)
        mini_compress_kv = _kv_storage.view(mini_num_blocks, self.block_size, 1, self.kv_dim)
        mini_compress_rope = _rope_storage.view(mini_num_blocks, self.block_size, 1, self.rope_dim)
        sel_kv = _kv_storage.view(sel_blocks, self.compress_block_size, self.kv_dim)
        sel_rope = _rope_storage.view(sel_blocks, self.compress_block_size, self.rope_dim)

        # gather-side: rows in the 64-group view (flat batch*topk+k)
        sel_block_table = torch.arange(
            sel_blocks, dtype=torch.int32, device=self.device,
        ).view(self.max_batch_size, topk)

        # attn_op-side (Phase B2): batch b owns blocks [b*(topk/gpb), b*(topk/gpb)+topk/gpb)
        # in mini_compress_kv's 128-block layout.
        mini_cmp_block_table = torch.arange(
            mini_num_blocks, dtype=torch.int32, device=self.device,
        ).view(self.max_batch_size, topk // gpb)

        # attn_op-side sparse_indices: each batch attends to local groups [0..topk-1]
        mini_sparse_indices = (
            torch.arange(topk, dtype=torch.int32, device=self.device)
            .view(1, 1, 1, topk)
            .expand(self.max_batch_size, 1, 1, topk)
            .contiguous()
        )

        # When topk > TOPK_SPLIT_NUM, gather() splits into chunks of ≤32.
        n_splits = (topk + TOPK_SPLIT_NUM - 1) // TOPK_SPLIT_NUM
        sel_block_status_list = []
        for s in range(n_splits):
            chunk = min(TOPK_SPLIT_NUM, topk - s * TOPK_SPLIT_NUM)
            sel_block_status_list.append(torch.full(
                (self.max_batch_size, 1, 1, chunk + 1),
                -1, dtype=torch.int32, device=self.device,
            ))

        state.sel_kv_cache = sel_kv
        state.sel_k_rope = sel_rope
        state.sel_block_table = sel_block_table
        state.sel_block_status = sel_block_status_list[0]
        state.sel_block_status_list = sel_block_status_list
        state.local_topk = topk
        # B2-specific attributes (attn_op reads these when TIDALCACHE_ATTN_ON_SEL=1)
        state.mini_compress_kv = mini_compress_kv
        state.mini_compress_rope = mini_compress_rope
        state.mini_cmp_block_table = mini_cmp_block_table
        state.mini_sparse_indices = mini_sparse_indices

    def alloc_layer(self, layer_name: str,
                    local_topk: int | None = None) -> LayerOffloadState:
        """Ensure both Host and Device Selection sides are allocated.

        If layer's Host side already exists (from an earlier scatter call),
        just fills in the sel_* side using local_topk. This ordering keeps
        sel_kv sized correctly for CP mode (topk = index_topk // cp_size).

        Args:
            local_topk: Actual per-rank topk. Defaults to self.index_topk.
        """
        state = self.ensure_host_allocated(layer_name)
        if state.sel_kv_cache is not None:
            return state

        topk = local_topk if local_topk is not None else self.index_topk
        self._alloc_sel_side(state, topk)

        logger.info(
            "Layer %s: sel side allocated (local_topk=%d)",
            layer_name, topk,
        )
        self._alloc_count += 1
        # Emit rolling HBM summary — grep the last one for post-alloc state.
        self._log_hbm_rank0(f"after_alloc_layer[{self._alloc_count}]")
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

        When topk > 32, the CANN operator's vector path requires
        selection_topk_block_size=1 which conflicts with group-level
        indices. We split into chunks of ≤32 to stay on the scalar path.
        """
        state = self.layers[layer_name]
        # Auto-upgrade: if scatter (PATCH3) alloc'd host only, allocate sel
        # side now that we know the real local_topk from topk_indices shape.
        if state.sel_kv_cache is None:
            actual_topk = topk_indices.shape[-1]
            self._alloc_sel_side(state, actual_topk)
            logger.info(
                "Layer %s: sel side auto-upgraded at gather (local_topk=%d)",
                layer_name, actual_topk,
            )
            self._alloc_count += 1
            self._log_hbm_rank0(f"after_alloc_layer[{self._alloc_count}]")

        gw = self._get_gather_wrapper()

        batch_size = full_block_table.shape[0]
        topk_indices = topk_indices[:batch_size]
        full_actual_seq = full_actual_seq[:batch_size]
        full_q_actual_seq = full_q_actual_seq[:batch_size]

        # Derive topk from the actual topk_indices shape. state.local_topk is
        # the alloc'd max — may differ if the layer was scatter-alloc'd with
        # a different topk assumption.
        topk = topk_indices.shape[-1]
        n_splits = (topk + TOPK_SPLIT_NUM - 1) // TOPK_SPLIT_NUM
        assert n_splits <= len(state.sel_block_status_list), (
            f"gather n_splits={n_splits} exceeds alloc'd block_status_list "
            f"of size {len(state.sel_block_status_list)} for {layer_name}"
        )

        logger.debug(
            "[GATHER] %s batch=%d topk=%d splits=%d state.local_topk=%s",
            layer_name, batch_size, topk, n_splits, state.local_topk,
        )
        # Diagnostic: on rank 0, first-decode-step log block_status stats to
        # confirm reset behavior and detect stale entries.
        if os.environ.get('TIDALCACHE_LOG_GATHER', '0') == '1':
            try:
                _bs0 = state.sel_block_status_list[0][:batch_size]
                _neg = (_bs0 < 0).sum().item()
                _tot = _bs0.numel()
                _first = _bs0.flatten()[:min(8, _tot)].tolist()
                logger.info(
                    "[GATHER-STATS] %s batch=%d topk=%d "
                    "block_status[0]: total=%d, invalid(-1)=%d, first_%d=%s",
                    layer_name, batch_size, topk,
                    _tot, _neg, len(_first), _first,
                )
            except Exception:
                pass

        # B2 correctness fix (2026-09-21): force fresh gather every call.
        # `sel_block_status` is a kernel-level cache tracking which blocks are
        # currently in `sel_kv`. reset_requests()/reset_all() are defined but
        # never called from anywhere — so status leaks across requests. When a
        # batch slot is reused, kernel may see "cached" == new block-id and
        # skip the copy, serving stale data. Symptom: long-context corruption
        # and, downstream, NPU vector-core faults.
        #
        # Default: force fresh copy (fill -1 before each chunk). Set
        # `TIDALCACHE_GATHER_CACHE=1` to opt back into the cached path
        # (only when you know reset is being called properly).
        import os as _tc_os_g
        _tc_gather_cache = _tc_os_g.environ.get('TIDALCACHE_GATHER_CACHE', '0') == '1'

        sel_actual_seq = None
        for s in range(n_splits):
            k_start = s * TOPK_SPLIT_NUM
            k_end = min(k_start + TOPK_SPLIT_NUM, topk)
            chunk_k = k_end - k_start

            chunk_indices = topk_indices[:, :, :, k_start:k_end].contiguous()
            chunk_bt = state.sel_block_table[:batch_size, k_start:k_end].contiguous()
            chunk_bs = state.sel_block_status_list[s][:batch_size]
            if not _tc_gather_cache:
                chunk_bs.fill_(-1)  # invalidate → kernel must copy

            sel_actual_seq = gw.npu_gather_selection_kv_cache(
                state.sel_k_rope,
                state.sel_kv_cache,
                chunk_bt,
                chunk_bs,
                chunk_indices,
                state.npu_k_rope,
                state.npu_kv_cache,
                full_block_table,
                full_actual_seq,
                full_q_actual_seq,
                self.compress_block_size,
            )

        logger.debug("[GATHER] %s done", layer_name)
        return state.sel_kv_cache, state.sel_k_rope, sel_actual_seq

    # ── Batch Lifecycle ──

    def reset_requests(self, layer_name: str, batch_indices: torch.Tensor):
        """Reset block_status for finished/new requests."""
        state = self.layers[layer_name]
        for bs in state.sel_block_status_list:
            for idx in batch_indices.tolist():
                bs[idx].fill_(-1)

    def reset_all(self, layer_name: str):
        """Reset all batch slots for a layer."""
        state = self.layers[layer_name]
        for bs in state.sel_block_status_list:
            bs.fill_(-1)

    # ── Cleanup ──

    def cleanup(self):
        """Release all hugepage allocations, NPU registrations, and unlink files.

        Robust to B3 v2 fast-path state (mmap_kv=None, fd_kv=-1, path_kv=""),
        idempotent, and never raises. Also unlinks any file we recorded in
        `_created_hugepage_paths` that survived the per-state pass — protects
        against leaks when state fields were not populated (partial init crash).
        """
        # 1) Per-layer state: unregister from NPU, close mmap+fd, unlink files
        try:
            zcn = self._get_zero_copy()
        except Exception:
            zcn = None
        for name, state in list(self.layers.items()):
            for host_tensor_attr in ("host_kv_cache", "host_k_rope"):
                if zcn is None:
                    break
                t = getattr(state, host_tensor_attr, None)
                # Skip vllm-owned compress view — we didn't register it.
                if t is None or host_tensor_attr == "host_kv_cache" and state.mmap_kv is None:
                    continue
                try:
                    zcn.unregister_host(t)
                except Exception as e:
                    logger.debug("unregister %s/%s: %s", name, host_tensor_attr, e)
            for mmap_attr, fd_attr, path_attr in (
                ("mmap_kv", "fd_kv", "hugepage_path_kv"),
                ("mmap_rope", "fd_rope", "hugepage_path_rope"),
            ):
                m = getattr(state, mmap_attr, None)
                fd = getattr(state, fd_attr, -1)
                path = getattr(state, path_attr, "")
                try:
                    if m is not None:
                        m.close()
                except Exception:
                    pass
                try:
                    if fd is not None and fd >= 0:
                        os.close(fd)
                except Exception:
                    pass
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    self._created_hugepage_paths.discard(path)
        self.layers.clear()

        # 2) Group allocations (B3 v2 compress group from alloc_group_host_tensor)
        for grp in list(getattr(self, "_group_allocations", []) or []):
            try:
                if zcn is not None and grp.get("host") is not None:
                    try:
                        zcn.unregister_host(grp["host"])
                    except Exception:
                        pass
                m = grp.get("mmap")
                if m is not None:
                    try:
                        m.close()
                    except Exception:
                        pass
                fd = grp.get("fd", -1)
                if fd is not None and fd >= 0:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                path = grp.get("path", "")
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    self._created_hugepage_paths.discard(path)
            except Exception as e:
                logger.debug("group cleanup: %s", e)
        if hasattr(self, "_group_allocations"):
            self._group_allocations.clear()

        # 3) Safety net: unlink any file we created but didn't remove above
        for path in list(self._created_hugepage_paths):
            try:
                os.unlink(path)
            except OSError:
                pass
        self._created_hugepage_paths.clear()

        logger.info("TidalCache cleanup complete (pid=%d)", os.getpid())

    def __del__(self):
        # __del__ runs during interpreter shutdown; be defensive.
        try:
            self.cleanup()
        except Exception:
            pass
