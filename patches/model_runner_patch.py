"""
Patch for vllm_ascend/worker/model_runner_v1.py

Target: NPUModelRunner class
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
# ═══════════════════════════════════════════════════════════

PATCH_2_INIT_KV = '''
        # ── TidalCache: allocate Host KV + Selection Cache ──
        if self.kv_offload_enabled and self.use_sparse:
            from tidalcache.offload_manager import TidalCacheManager

            # Read DSA parameters from model config
            hf_config = self.model_config.hf_text_config
            index_topk = getattr(hf_config, 'index_topk', 4)
            # MLA dimensions
            kv_lora_rank = getattr(hf_config, 'kv_lora_rank', 512)
            qk_rope_head_dim = getattr(hf_config, 'qk_rope_head_dim', 64)
            block_size = 64  # DSA block size

            self._tidalcache_mgr = TidalCacheManager(
                num_blocks=kv_cache_config.num_blocks,
                block_size=block_size,
                kv_dim=kv_lora_rank,       # 512 for DeepSeek-V3
                rope_dim=qk_rope_head_dim, # 64
                index_topk=index_topk,
                max_batch_size=self.scheduler_config.max_num_seqs,
                dtype=self.model_config.dtype,
                device=self.device,
            )

            # Allocate per-layer and bind to DSA impl
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

                self._tidalcache_mgr.alloc_layer(layer_name)
                impl._tidalcache_mgr = self._tidalcache_mgr

            logger.info(
                "TidalCache: initialized %d layers, topk=%d, blocks=%d",
                len(self._tidalcache_mgr.layers), index_topk,
                kv_cache_config.num_blocks,
            )
'''
