"""
Patch for vllm_ascend/attention/dsa_v1.py

This file documents the exact modifications needed. Apply manually or use as reference.

Target: vllm_ascend/attention/dsa_v1.py in vllm-ascend repo
"""

# ═══════════════════════════════════════════════════════════
# PATCH 1: AscendDSAImpl.__init__() — add TidalCache attributes
# Insert after line ~1452 (after self.skip_topk assignment)
# ═══════════════════════════════════════════════════════════

PATCH_1_INIT = '''
        # ── TidalCache: KV offload ──
        from tidalcache import TIDALCACHE_ENABLED
        self.kv_offload_enabled = TIDALCACHE_ENABLED
        self._tidalcache_mgr = None  # set externally by model_runner
'''

# ═══════════════════════════════════════════════════════════
# PATCH 2: _forward_decode() — insert GatherSelectionKvCache
# Insert after line ~2366 (after _update_indexcache_topk_indices)
# Before line ~2368 (before attn_op = ...)
# ═══════════════════════════════════════════════════════════

PATCH_2_GATHER = '''
            # ── TidalCache: Sparse Host→Device Gather ──
            if self.kv_offload_enabled and self._tidalcache_mgr is not None:
                B = hidden_states.shape[0]
                sel_kv, sel_rope, sel_actual_seq = self._tidalcache_mgr.gather(
                    layer_name=layer_name,
                    topk_indices=compress_topk_idxs.view(B, 1, 1, self.index_topk),
                    full_block_table=compressor_decode_metadata.block_table,
                    full_actual_seq=actual_seq_lengths_key,
                    full_q_actual_seq=actual_seq_lengths_query,
                )
                # Replace compress_kv_cache with gathered selection cache
                compress_kv_cache = sel_kv
'''

# ═══════════════════════════════════════════════════════════
# PATCH 3: _forward_decode() — compressor scatter to Host
# Modify line ~2330-2331 (the compress scatter call)
# ═══════════════════════════════════════════════════════════

PATCH_3_SCATTER_ORIGINAL = '''
            if compressed_kv.shape[0] > 0:
                DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
'''

PATCH_3_SCATTER_MODIFIED = '''
            if compressed_kv.shape[0] > 0:
                if self.kv_offload_enabled and self._tidalcache_mgr is not None:
                    # TidalCache: scatter to Host Full KV (via NPU view tensor)
                    host_kv = self._tidalcache_mgr.layers[layer_name].npu_kv_cache
                    DeviceOperator.dsa_kv_compress_scatter(host_kv, compressed_kv, compress_slot_mapping)
                else:
                    DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
'''

# ═══════════════════════════════════════════════════════════
# PATCH 4: _forward_decode() — attn_op call with selection block_table
# Modify line ~2389-2410 (compress_ratio == 4 branch)
# ═══════════════════════════════════════════════════════════

PATCH_4_ATTN_ORIGINAL = '''
        elif self.compress_ratio == 4:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=compressor_decode_metadata.block_table,
'''

PATCH_4_ATTN_MODIFIED = '''
        elif self.compress_ratio == 4:
            if self.kv_offload_enabled and self._tidalcache_mgr is not None:
                _cmp_block_table = self._tidalcache_mgr.layers[
                    layer_name].sel_block_table[:hidden_states.shape[0]]
            else:
                _cmp_block_table = compressor_decode_metadata.block_table
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=_cmp_block_table,
'''
