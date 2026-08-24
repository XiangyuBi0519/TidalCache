"""
Patch for vllm_ascend/worker/model_runner_v1.py

Target: NPUModelRunner class

Supports:
  - DeepSeek-V3 (DSA): all sparse layers get offload
  - DeepSeek-V4 (CSA/HCA hybrid): only CSA layers get offload,
    HCA layers are skipped (dense attention, no top-k selection)
"""

# ═══════════════════════════════════════════════════════════
# PATCH 1: __init__() — add TidalCache manager
# Insert after self.use_sparse is set
# ═══════════════════════════════════════════════════════════

PATCH_1_INIT = '''
        # ── TidalCache ──
        from tidalcache import TIDALCACHE_ENABLED
        self.kv_offload_enabled = TIDALCACHE_ENABLED
        self._tidalcache_mgr = None
'''

# ═══════════════════════════════════════════════════════════
# PATCH 2: initialize_kv_cache_tensors() — init TidalCache after KV allocation
# Insert after line ~4052 (after setting .kv_cache on static_forward_context)
# Before the return statement
#
# V4 compatibility notes:
#   - hf_config.attention_type per layer: "csa" | "hca" | "dsa"
#     Only "csa" and "dsa" layers use Lightning Indexer → TidalCache
#   - V4 uses FP8 for KV dims, BF16 for RoPE dims (mixed storage)
#     When hf_config has 'kv_cache_fp8=True', we set kv_dtype=float8_e4m3fn
#   - block_size may differ from hardcoded 64 in V4
#     Read from hf_config.compress_block_size if available
# ═══════════════════════════════════════════════════════════

PATCH_2_INIT_KV = '''
        # ── TidalCache: allocate Host KV + Selection Cache ──
        if self.kv_offload_enabled and self.use_sparse:
            from tidalcache.offload_manager import TidalCacheManager

            hf_config = self.model_config.hf_text_config
            index_topk = getattr(hf_config, 'index_topk', 4)
            kv_lora_rank = getattr(hf_config, 'kv_lora_rank', 512)
            qk_rope_head_dim = getattr(hf_config, 'qk_rope_head_dim', 64)
            block_size = getattr(hf_config, 'compress_block_size', 64)

            # V4 mixed precision: FP8 for KV, BF16 for RoPE
            import torch
            kv_dtype = self.model_config.dtype
            rope_dtype = self.model_config.dtype
            if getattr(hf_config, 'kv_cache_fp8', False):
                kv_dtype = torch.float8_e4m3fn
                rope_dtype = torch.bfloat16

            self._tidalcache_mgr = TidalCacheManager(
                num_blocks=kv_cache_config.num_blocks,
                block_size=block_size,
                kv_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                index_topk=index_topk,
                max_batch_size=self.scheduler_config.max_num_seqs,
                dtype=kv_dtype,
                device=self.device,
                rope_dtype=rope_dtype,
            )

            # Per-layer attention type (V4 hybrid: csa/hca interleaved)
            # Only allocate for layers with sparse selection (DSA/CSA)
            attn_types = getattr(hf_config, 'attention_types', None)
            num_layers_allocated = 0

            for layer_name in kv_caches:
                ctx = self.compilation_config.static_forward_context.get(layer_name)
                if ctx is None:
                    continue
                dsa_attn = getattr(ctx, 'dsa_attn', None)
                if dsa_attn is None:
                    continue
                impl = getattr(dsa_attn, 'impl', None)
                if impl is None or not getattr(impl, 'kv_offload_enabled', False):
                    continue

                # V4: skip HCA layers (dense attention, no top-k)
                if attn_types is not None:
                    layer_idx = self._extract_layer_idx(layer_name)
                    if layer_idx is not None:
                        layer_type = self._get_layer_attn_type(
                            attn_types, layer_idx)
                        if layer_type == 'hca':
                            logger.info(
                                "TidalCache: skip HCA layer %s", layer_name)
                            impl.kv_offload_enabled = False
                            continue

                self._tidalcache_mgr.alloc_layer(layer_name)
                impl._tidalcache_mgr = self._tidalcache_mgr
                num_layers_allocated += 1

            logger.info(
                "TidalCache: initialized %d layers (skipped HCA), "
                "topk=%d, blocks=%d, kv_dtype=%s, rope_dtype=%s",
                num_layers_allocated, index_topk,
                kv_cache_config.num_blocks, kv_dtype, rope_dtype,
            )
'''

# ═══════════════════════════════════════════════════════════
# PATCH 3: Helper methods for V4 layer type detection
# Add to NPUModelRunner class
# ═══════════════════════════════════════════════════════════

PATCH_3_HELPERS = '''
    @staticmethod
    def _extract_layer_idx(layer_name: str) -> int | None:
        """Extract layer index from name like 'model.layers.3.self_attn'."""
        parts = layer_name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    @staticmethod
    def _get_layer_attn_type(attn_types, layer_idx: int) -> str:
        """Resolve attention type for a layer index.

        attn_types formats (from hf_config):
          V4: {"csa": [...layer indices...], "hca": [...layer indices...]}
          or list-of-tuples: [("csa", count), ("hca", count), ...]
        Returns "csa", "hca", "dsa", or "unknown".
        """
        if isinstance(attn_types, dict):
            for atype, indices in attn_types.items():
                if layer_idx in indices:
                    return atype
            return "unknown"
        if isinstance(attn_types, (list, tuple)):
            pos = 0
            for atype, count in attn_types:
                if pos <= layer_idx < pos + count:
                    return atype
                pos += count
            return "unknown"
        return "unknown"
'''
