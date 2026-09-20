#!/usr/bin/env python3
"""Apply TidalCache patches to vllm-ascend source files.

Usage:
    python3 apply_patches.py /path/to/vllm-ascend          # apply
    python3 apply_patches.py /path/to/vllm-ascend --check   # dry-run
    python3 apply_patches.py /path/to/vllm-ascend --rollback # restore .bak

Pattern-based: works regardless of vllm-ascend version.
"""

import sys
import os
import shutil
import re

MARKER = "# ── TidalCache"


def patch_file(path, patches, dry_run=False):
    with open(path, "r") as f:
        content = f.read()

    if MARKER in content:
        print(f"  SKIP (already patched): {path}")
        return False

    for name, (anchor, insertion, mode) in patches.items():
        if anchor not in content:
            print(f"  ERROR: anchor not found for {name}")
            print(f"    Expected: {anchor[:80]}...")
            sys.exit(1)

        if mode == "after":
            content = content.replace(anchor, anchor + insertion, 1)
        elif mode == "replace":
            content = content.replace(anchor, insertion, 1)
        print(f"  {name}: OK")

    if not dry_run:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"  Backed up: {bak}")
        with open(path, "w") as f:
            f.write(content)
        print(f"  Written: {path}")
    else:
        print(f"  (dry-run, not written)")

    return True


# ═══════════════════════════════════════════
# dsa_v1.py patches
# ═══════════════════════════════════════════

DSA_PATCHES = {
    "PATCH1_init": (
        # Anchor: end of __init__, after use_index_cache assignment
        '''        self.use_index_cache = self.skip_topk or getattr(
            self.vllm_config.model_config.hf_config,
            "use_index_cache",
            False,
        )''',
        '''

        # ── TidalCache: KV offload ──
        from tidalcache import TIDALCACHE_ENABLED
        self.kv_offload_enabled = TIDALCACHE_ENABLED
        self._tidalcache_mgr = None''',
        "after",
    ),

}

# PATCH2: insert gather BEFORE attn_op (outside use_index_cache if block)
# Use regex to find attn_op line in the decode path
PATCH2_GATHER_CODE = '''
        # ── TidalCache: lazy init + Sparse Host→Device Gather ──
        # Define _tc_cmp_block_table for PATCH4 safely: compressor_decode_metadata
        # is only assigned in the compress_ratio==4/128 branches, so use
        # locals().get() to avoid UnboundLocalError when neither branch ran.
        _cdm = locals().get('compressor_decode_metadata')
        _tc_cmp_block_table = _cdm.block_table if _cdm is not None else None
        # Save reference to vllm's compress_kv_cache BEFORE any rebind so a
        # later compress-poison test can target the original tensor.
        _orig_compress_ref = compress_kv_cache
        if self.kv_offload_enabled and _cdm is not None:
            if self._tidalcache_mgr is None:
                import tidalcache as _tc
                self._tidalcache_mgr = _tc._GLOBAL_MANAGER
            if self._tidalcache_mgr is not None:
                import torch as _torch
                import logging as _logging
                import os as _tcos_b2
                _tclog = _logging.getLogger("tidalcache")
                B = hidden_states.shape[0]
                _local_topk = compress_topk_idxs.numel() // B
                if layer_name not in self._tidalcache_mgr.layers:
                    self._tidalcache_mgr.alloc_layer(layer_name, local_topk=_local_topk)
                _sel_kv, _sel_rope, _sel_actual_seq = self._tidalcache_mgr.gather(
                    layer_name=layer_name,
                    topk_indices=compress_topk_idxs.view(B, 1, 1, _local_topk),
                    full_block_table=compressor_decode_metadata.block_table,
                    full_actual_seq=actual_seq_lengths_key,
                    full_q_actual_seq=_torch.ones(B, dtype=_torch.int32, device=hidden_states.device),
                )
                # ── TidalCache validation switch: poison gather output ──
                _poison_mode = _tcos_b2.environ.get("TIDALCACHE_POISON", "0")
                if _poison_mode == "1":
                    _sel_kv.fill_(-1000.0)
                    _sel_rope.fill_(-1000.0)
                    _tclog.info("[POISON] %s gather output filled with -1000", layer_name)

                _state = self._tidalcache_mgr.layers[layer_name]
                _attn_on_sel = _tcos_b2.environ.get("TIDALCACHE_ATTN_ON_SEL", "0") == "1"
                _max_batch = _state.mini_cmp_block_table.shape[0]
                # attn_op's query B is the number of REQUESTS, which equals the
                # original block_table's dim 0. hidden_states.shape[0] counts
                # tokens (can be > requests during warmup / chunked prefill).
                _actual_reqs = _cdm.block_table.shape[0]
                if _attn_on_sel and _state.sel_kv_cache is not None and _actual_reqs <= _max_batch:
                    # Rebind local vars flowing into attn_op. Match the shape/
                    # rank of the ORIGINAL compress_topk_idxs so SparseAttnSharedkv's
                    # TND layout parser identifies the N axis correctly.
                    compress_kv_cache = _state.mini_compress_kv
                    _orig_shape = tuple(compress_topk_idxs.shape)
                    _topk_actual = _orig_shape[-1]
                    _flat_idx = _torch.arange(
                        _topk_actual,
                        dtype=compress_topk_idxs.dtype,
                        device=hidden_states.device,
                    )
                    _bcast_shape = [1] * (len(_orig_shape) - 1) + [_topk_actual]
                    compress_topk_idxs = _flat_idx.view(*_bcast_shape).expand(*_orig_shape).contiguous()
                    _tc_cmp_block_table = _state.mini_cmp_block_table[:_actual_reqs]
                    if not getattr(self, '_tc_logged_b2_' + layer_name.replace('.','_'), False):
                        _tclog.info(
                            '[ATTN-ON-SEL first] %s → attn reads sel_kv (Phase B2), reqs=%d, orig_shape=%s',
                            layer_name, _actual_reqs, _orig_shape,
                        )
                        setattr(self, '_tc_logged_b2_' + layer_name.replace('.','_'), True)
                    _tclog.debug("[ATTN-ON-SEL] %s B=%d reqs=%d", layer_name, B, _actual_reqs)
                else:
                    # Phase B1: copy-back into compress_kv_cache, attn reads original.
                    _cbs = self._tidalcache_mgr.compress_block_size  # 64
                    _gpb = compress_kv_cache.shape[1] // _cbs
                    _bt_full = compressor_decode_metadata.block_table
                    _aB = min(B, _bt_full.shape[0])
                    _tidx = compress_topk_idxs.view(B, _local_topk)[:_aB]
                    _bseq = (_tidx // _gpb).long()
                    _goff = (_tidx % _gpb).long()
                    _cbt = _bt_full[:_aB].long()
                    _pblk = _torch.gather(_cbt, 1, _bseq)
                    _dst = (_pblk * _gpb + _goff).view(-1)
                    _n = _aB * _local_topk
                    _trail = compress_kv_cache.shape[2:]
                    _cmp64 = compress_kv_cache.view(-1, _cbs, *_trail)
                    _src = _sel_kv[:_n]
                    if _src.shape[2:] != _trail:
                        _src = _src.view(_n, _cbs, *_trail)
                    _cmp64[_dst] = _src
                    _tclog.debug("[COPYBACK] %s B=%d aB=%d topk=%d dst_blocks=%d", layer_name, B, _aB, _local_topk, _n)

                # ── Step 1 test: poison vllm's compress_kv_cache before attn ──
                # TIDALCACHE_POISON_COMPRESS=1 fills the ORIGINAL vllm-allocated
                # Device compress_kv_cache with NaN AFTER gather/rebind. Under
                # Phase B2 attn reads sel_kv (aliased to mini_compress_kv), so
                # the NaN in the vllm tensor should be invisible to attention
                # if nothing else still reads compress_kv_cache. If output stays
                # correct → safe to release the vllm tensor storage in B3.
                if _tcos_b2.environ.get("TIDALCACHE_POISON_COMPRESS", "0") == "1":
                    _orig_compress_ref.zero_()
                    if not getattr(self, '_tc_poison_c_' + layer_name.replace('.','_'), False):
                        _tclog.info(
                            '[POISON-COMPRESS first] %s → vllm compress_kv_cache zeroed',
                            layer_name,
                        )
                        setattr(self, '_tc_poison_c_' + layer_name.replace('.','_'), True)
'''

# PATCH3: scatter redirect — replace dsa_kv_compress_scatter target
# Use regex: find the scatter call with compress_kv_cache as first arg (decode path)
# Capture the remaining args so the replacement preserves them exactly.

# PATCH4: removed — with the copy-back approach, attn_op uses original
# compress_kv_cache and block_table unchanged.


# ═══════════════════════════════════════════
# model_runner_v1.py patches
# ═══════════════════════════════════════════

# Find a stable anchor near use_sparse for PATCH1
# We look for the attn_backend line that comes after use_sparse setup
MR_PATCHES = {}

# PATCH1: add TidalCache flag in __init__
# Anchor: "self.attn_backend = get_attn_backend(" which is stable across versions
MR_PATCH1_ANCHOR = '''        self.attn_backend = get_attn_backend('''
MR_PATCH1_INSERT = '''
        # ── TidalCache ──
        from tidalcache import TIDALCACHE_ENABLED
        self.kv_offload_enabled = TIDALCACHE_ENABLED
        self._tidalcache_mgr = None

'''
MR_PATCHES["PATCH1_init"] = (MR_PATCH1_ANCHOR, MR_PATCH1_INSERT + MR_PATCH1_ANCHOR, "replace")

# PATCH2_INIT: create TidalCache manager EARLY — before _allocate_kv_cache_tensors
# runs. Needed so MR_PATCH3's Host-backed compress replacement (which fires
# inside _allocate_kv_cache_tensors) can find the manager.
# Anchor: 'kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)'
MR_PATCH2_INIT_CODE = '''
        # ── TidalCache: create manager BEFORE vllm allocation, so MR_PATCH3
        # (Host-backed compress replacement) has access to it.
        import os as _tcos_i
        import torch as _torch_i
        _hf_cfg_i = getattr(self.model_config, 'hf_text_config', None)
        _has_topk_i = _hf_cfg_i is not None and hasattr(_hf_cfg_i, 'index_topk')
        if self.kv_offload_enabled and _has_topk_i:
            import tidalcache as _tc_i
            from tidalcache.offload_manager import TidalCacheManager as _TCM_i

            _index_topk_i = getattr(_hf_cfg_i, 'index_topk', 512)
            _kv_dim_i = getattr(_hf_cfg_i, 'kv_lora_rank', None)
            if _kv_dim_i is None:
                _kv_dim_i = getattr(_hf_cfg_i, 'head_dim', 512)
            _qk_rope_i = getattr(_hf_cfg_i, 'qk_rope_head_dim', 64)
            _cbs_i = getattr(_hf_cfg_i, 'compress_block_size', 64)
            try:
                _grp_i = kv_cache_config.kv_cache_groups[0]
                _spec_i = list(_grp_i.kv_cache_spec.values())[0]
                _blk_i = _spec_i.block_size
            except (AttributeError, IndexError, KeyError):
                _blk_i = self.cache_config.block_size
            _kv_dtype_i = self.model_config.dtype
            _rope_dtype_i = self.model_config.dtype
            if getattr(_hf_cfg_i, 'kv_cache_fp8', False):
                _kv_dtype_i = _torch_i.float8_e4m3fn
                _rope_dtype_i = _torch_i.bfloat16
            self._tidalcache_mgr = _TCM_i(
                num_blocks=kv_cache_config.num_blocks,
                block_size=_blk_i,
                kv_dim=_kv_dim_i,
                rope_dim=_qk_rope_i,
                index_topk=_index_topk_i,
                max_batch_size=self.scheduler_config.max_num_seqs,
                dtype=_kv_dtype_i,
                device=self.device,
                rope_dtype=_rope_dtype_i,
                compress_block_size=_cbs_i,
            )
            _tc_i._GLOBAL_MANAGER = self._tidalcache_mgr
            logger.info(
                "TidalCache: manager created EARLY (before _allocate_kv_cache_tensors) "
                "topk=%d blocks=%d kv_dim=%d",
                _index_topk_i, kv_cache_config.num_blocks, _kv_dim_i,
            )

'''

# PATCH2: initialize TidalCache in initialize_kv_cache_tensors — HBM log at end
# Anchor: "return kv_caches" at the end of initialize_kv_cache_tensors
MR_PATCH2_CODE = '''
        # ── TidalCache: (manager already created above; skip if already done) ──
        import os as _tcos
        import torch as _torch
        _hf_cfg = getattr(self.model_config, 'hf_text_config', None)
        _has_topk = _hf_cfg is not None and hasattr(_hf_cfg, 'index_topk')
        if self.kv_offload_enabled and _has_topk and self._tidalcache_mgr is None:
            import tidalcache
            from tidalcache.offload_manager import TidalCacheManager

            hf_config = _hf_cfg
            index_topk = getattr(hf_config, 'index_topk', 512)
            kv_dim = getattr(hf_config, 'kv_lora_rank', None)
            if kv_dim is None:
                kv_dim = getattr(hf_config, 'head_dim', 512)
            qk_rope_head_dim = getattr(hf_config, 'qk_rope_head_dim', 64)
            compress_block_size = getattr(hf_config, 'compress_block_size', 64)
            try:
                _grp = kv_cache_config.kv_cache_groups[0]
                _spec = list(_grp.kv_cache_spec.values())[0]
                kv_block_size = _spec.block_size
            except (AttributeError, IndexError, KeyError):
                kv_block_size = self.cache_config.block_size

            kv_dtype = self.model_config.dtype
            rope_dtype = self.model_config.dtype
            if getattr(hf_config, 'kv_cache_fp8', False):
                kv_dtype = _torch.float8_e4m3fn
                rope_dtype = _torch.bfloat16

            self._tidalcache_mgr = TidalCacheManager(
                num_blocks=kv_cache_config.num_blocks,
                block_size=kv_block_size,
                kv_dim=kv_dim,
                rope_dim=qk_rope_head_dim,
                index_topk=index_topk,
                max_batch_size=self.scheduler_config.max_num_seqs,
                dtype=kv_dtype,
                device=self.device,
                rope_dtype=rope_dtype,
                compress_block_size=compress_block_size,
            )
            tidalcache._GLOBAL_MANAGER = self._tidalcache_mgr

            logger.info(
                "TidalCache: manager created, topk=%d, blocks=%d, "
                "kv_dim=%d, kv_block_size=%d, compress_block_size=%d",
                index_topk, kv_cache_config.num_blocks, kv_dim,
                kv_block_size, compress_block_size,
            )

        # ── TidalCache Phase B3.1: physical tensor replacement (same size) ──
        # Replace kv_caches[layer_name][0] with a fresh same-shape tensor for
        # each DSA layer, then update self.kv_caches list and forward_context
        # binding so ALL downstream reads (including any captured CUDA graph)
        # see the new tensor instead of vllm's original allocation.
        #
        # Purpose (B3.1): validate the replacement mechanism end-to-end. Same
        # size = no HBM savings yet, but subsequent phases (B3.2 poison the
        # NEW tensor to prove nothing reads it; B3.3 shrink) build on this.
        #
        # Gated by TIDALCACHE_REPLACE_KV=1. Requires kv_offload_enabled.
        # ── Phase B3 planning: dump kv_cache_groups structure (rank 0 only) ──
        # Decides whether compress (self_attn.attn) and indexer (self_attn.indexer.k_cache)
        # are in the SAME group (bad: they share raw_tensor → can't shrink independently)
        # or DIFFERENT groups (good: can modify compress group's num_blocks in isolation).
        if getattr(self, 'rank', 0) == 0:
            import logging as _tc_lg_g
            _glog = _tc_lg_g.getLogger('tidalcache')
            try:
                _grps = kv_cache_config.kv_cache_groups
                _glog.info('[KV-GROUPS] total=%d, num_blocks=%d', len(_grps), kv_cache_config.num_blocks)
                for _gi, _grp in enumerate(_grps):
                    # group has kv_cache_spec (usually single spec) and layer_names
                    _spec = _grp.kv_cache_spec
                    if hasattr(_spec, 'values'):
                        _spec_vals = list(_spec.values())
                        _spec_str = f'{type(_spec_vals[0]).__name__}xN' if _spec_vals else 'empty'
                    else:
                        _spec_str = type(_spec).__name__
                    _layer_names = getattr(_grp, 'layer_names', [])
                    _sample = _layer_names[0] if _layer_names else '?'
                    _sample_suffix = _sample
                    _sp = _sample.split('.')
                    if len(_sp) >= 3 and _sp[0] == 'model' and _sp[1] == 'layers':
                        _sample_suffix = '.'.join(_sp[3:])
                    _glog.info(
                        '[KV-GROUPS] group[%d] spec=%s, num_layers=%d, sample_suffix=%s',
                        _gi, _spec_str, len(_layer_names), _sample_suffix,
                    )
                    # Also log the first few layers of this group to see prefixes
                    if len(_layer_names) <= 5:
                        for _ln in _layer_names:
                            _glog.info('  member: %s', _ln)
                    else:
                        for _ln in _layer_names[:3]:
                            _glog.info('  member: %s', _ln)
                        _glog.info('  ... (%d more)', len(_layer_names) - 3)
            except Exception as _e:
                _glog.warning('[KV-GROUPS] failed: %s', _e)

        # TIDALCACHE_REPLACE_KV modes:
        #   0 / unset : off
        #   dry       : diagnostic only — log kv_caches structure, don't touch
        #   1         : ACTUAL same-size replacement (only if free HBM allows)
        _replace_kv = _tcos.environ.get('TIDALCACHE_REPLACE_KV', '0').lower()
        if _replace_kv in ('dry', '1') and self.kv_offload_enabled and _has_topk:
            import logging as _tc_lg_r
            _rlog = _tc_lg_r.getLogger('tidalcache')
            _rank0 = getattr(self, 'rank', 0) == 0
            _replaced_count = 0
            _skipped_count = 0
            _dsa_candidates = 0

            # First pass (rank 0 only): group layer names by suffix pattern to
            # understand the kv_caches layout without dumping 168 log lines.
            def _tc_suf(_n):
                # Strip 'model.layers.N.' prefix without regex (avoids escape
                # sequence warnings inside the outer triple-quoted patch string).
                _p = _n.split('.')
                if len(_p) >= 3 and _p[0] == 'model' and _p[1] == 'layers':
                    return '.'.join(_p[3:])
                return _n
            if _rank0:
                _suffix_stats = {}
                for _lname, _entry in kv_caches.items():
                    _suf = _tc_suf(_lname)
                    if _suf not in _suffix_stats:
                        _shape_desc = 'unknown'
                        if isinstance(_entry, (tuple, list)):
                            _parts = []
                            for _v in _entry:
                                if hasattr(_v, 'shape'):
                                    _parts.append(f'{_v.dtype}{tuple(_v.shape)}')
                                else:
                                    _parts.append(type(_v).__name__)
                            _shape_desc = f'{type(_entry).__name__}[{", ".join(_parts)}]'
                        elif hasattr(_entry, 'shape'):
                            _shape_desc = f'{_entry.dtype}{tuple(_entry.shape)}'
                        _suffix_stats[_suf] = [0, _shape_desc]
                    _suffix_stats[_suf][0] += 1
                _rlog.info(
                    '[REPLACE-KV suffix-stats] total_entries=%d unique_suffixes=%d',
                    len(kv_caches), len(_suffix_stats),
                )
                for _suf, (_cnt, _sd) in sorted(_suffix_stats.items()):
                    _rlog.info('[REPLACE-KV suffix] %s x%d → %s', _suf, _cnt, _sd)

            # DSA layer detection: a layer is DSA iff it has an indexer.k_cache
            # entry in kv_caches. The compress_kv_cache lives at the same layer's
            # 'self_attn.attn' suffix. Dense/HCA layers also have 'self_attn.attn'
            # but no indexer — those we must NOT touch or attention breaks.
            _dsa_prefixes = set()
            _INDEXER_SFX = '.self_attn.indexer.k_cache'
            _ATTN_SFX = '.self_attn.attn'
            for _lname in kv_caches:
                if _lname.endswith(_INDEXER_SFX):
                    _dsa_prefixes.add(_lname[:-len(_INDEXER_SFX)])
            if _rank0:
                _rlog.info('[REPLACE-KV] identified %d DSA layers', len(_dsa_prefixes))

            for _lname, _entry in list(kv_caches.items()):
                # Only replace DSA compress_kv_cache: '<prefix>.self_attn.attn'
                # where <prefix> also has an indexer.k_cache entry.
                if not _lname.endswith(_ATTN_SFX):
                    _skipped_count += 1
                    continue
                _prefix = _lname[:-len(_ATTN_SFX)]
                if _prefix not in _dsa_prefixes:
                    _skipped_count += 1
                    continue
                _tensor_first = None
                if isinstance(_entry, (tuple, list)) and len(_entry) >= 1:
                    if isinstance(_entry[0], _torch.Tensor) and _entry[0].dim() >= 3:
                        _tensor_first = _entry[0]
                elif isinstance(_entry, _torch.Tensor) and _entry.dim() >= 3:
                    _tensor_first = _entry
                if _tensor_first is None:
                    _skipped_count += 1
                    continue
                _dsa_candidates += 1

                if _replace_kv == 'dry':
                    if _rank0 and _dsa_candidates <= 3:
                        _rlog.info(
                            '[REPLACE-KV dry] candidate %s: shape=%s dtype=%s (would alloc %.2f MB)',
                            _lname, tuple(_tensor_first.shape), _tensor_first.dtype,
                            _tensor_first.numel() * _tensor_first.element_size() / 1024**2,
                        )
                    _replaced_count += 1
                    continue

                # WARNING: cannot free old storage — vllm allocates ONE big
                # raw_tensor per kv_cache_group, then slices it into multiple
                # views (compress + swa + state_cache etc). Calling
                # storage().resize_(0) on any view frees the SHARED storage
                # and breaks state_cache. Just allocate new independent
                # tensor; old view becomes orphaned but its shared storage
                # stays alive (still used by state_cache/swa/etc.).
                #
                # This means B3.1 same-size replacement DOUBLES memory
                # transiently. Fails with OOM if not enough free HBM.
                _old = _tensor_first
                _shape = tuple(_old.shape)
                _dtype = _old.dtype
                _dev = _old.device
                try:
                    _new = _torch.zeros(_shape, dtype=_dtype, device=_dev)
                except Exception as _e:
                    _rlog.error('[REPLACE-KV] OOM allocating %s: %s', _lname, _e)
                    break
                if isinstance(_entry, tuple):
                    _new_entry = (_new,) + tuple(_entry[1:])
                elif isinstance(_entry, list):
                    _new_entry = [_new] + list(_entry[1:])
                else:
                    _new_entry = _new
                kv_caches[_lname] = _new_entry
                for _i, _e in enumerate(self.kv_caches):
                    if _e is _entry:
                        self.kv_caches[_i] = _new_entry
                        break
                try:
                    _ctx = self.compilation_config.static_forward_context.get(_lname)
                    if _ctx is not None:
                        _ctx.kv_cache = [_new_entry]
                except Exception as _e:
                    _rlog.warning('[REPLACE-KV] forward_context update failed for %s: %s', _lname, _e)
                _replaced_count += 1
                if _rank0 and _replaced_count <= 2:
                    _rlog.info(
                        '[REPLACE-KV] %s: replaced (shape=%s, dtype=%s)',
                        _lname, _shape, _dtype,
                    )
            _rlog.info(
                '[REPLACE-KV %s] done: dsa_layers=%d, candidates=%d, replaced=%d, skipped=%d (rank=%d)',
                _replace_kv, len(_dsa_prefixes), _dsa_candidates, _replaced_count,
                _skipped_count, getattr(self, 'rank', 0),
            )

        # ── TidalCache: single-line HBM summary after vllm KV allocation ──
        # (per-layer TidalCache Device allocation is lazy; a second summary
        #  line will be emitted after all layers are allocated on first forward.)
        _tc_rank = getattr(self, 'rank', 0)
        if _tc_rank == 0:
            try:
                _tc_free, _tc_total = _torch.npu.mem_get_info()
                _GB = 1024 ** 3
                _tc_tag = _tcos.environ.get(
                    'TIDALCACHE_TAG',
                    'ON' if getattr(self, 'kv_offload_enabled', False) else 'OFF',
                )
                import logging as _tc_logging
                _tc_logging.getLogger('tidalcache').info(
                    "[HBM] after_vllm_kv_init [MODE=%s] | used=%.2f/%.2fGB | free=%.2fGB",
                    _tc_tag,
                    (_tc_total - _tc_free) / _GB,
                    _tc_total / _GB,
                    _tc_free / _GB,
                )
            except Exception as _e:
                logger.warning("TidalCache HBM logging failed: %s", _e)

        # ── B3 v2: register vllm's Host-backed compress views with TidalCache ──
        # After MR_PATCH3 swapped in Host-backed raw_tensor and vllm's reshape
        # split it into per-layer views, populate the manager's
        # _layer_compress_view map. TidalCache's per-layer ensure_host_allocated
        # will then reuse these views instead of allocating fresh hugepages —
        # solves the double-Host-alloc OOM.
        if (self.kv_offload_enabled and self._tidalcache_mgr is not None
                and _tcos.environ.get('TIDALCACHE_HOST_COMPRESS', '0') == '1'):
            _CMP_SFX = '.self_attn.attn'
            _IDX_SFX = '.self_attn.indexer.k_cache'
            # DSA layers = compress layers whose prefix also has an indexer entry
            _dsa_prefixes = set()
            for _lname in kv_caches:
                if _lname.endswith(_IDX_SFX):
                    _dsa_prefixes.add(_lname[:-len(_IDX_SFX)])
            _cbs_v = self._tidalcache_mgr.compress_block_size
            _kv_dim_v = self._tidalcache_mgr.kv_dim
            _registered = 0
            for _lname, _entry in kv_caches.items():
                if not _lname.endswith(_CMP_SFX):
                    continue
                _prefix = _lname[:-len(_CMP_SFX)]
                if _prefix not in _dsa_prefixes:
                    continue
                # Extract the compress tensor from the layer's kv_cache entry
                _cmp_t = None
                if isinstance(_entry, (tuple, list)) and len(_entry) >= 1:
                    if isinstance(_entry[0], _torch.Tensor):
                        _cmp_t = _entry[0]
                elif isinstance(_entry, _torch.Tensor):
                    _cmp_t = _entry
                if _cmp_t is None:
                    continue
                # Reshape to TidalCache's expected 3D layout
                # [num_blocks, block_size, N=1, kv_dim] → [num_blocks * gpb, cbs, kv_dim]
                try:
                    _view = _cmp_t.reshape(-1, _cbs_v, _kv_dim_v)
                except Exception as _e:
                    import logging as _lg_v
                    _lg_v.getLogger('tidalcache').warning(
                        '[HOST-COMPRESS] failed to reshape %s to (-1,%d,%d) — shape=%s: %s',
                        _lname, _cbs_v, _kv_dim_v, tuple(_cmp_t.shape), _e,
                    )
                    continue
                self._tidalcache_mgr.set_layer_compress_view(_lname, _view)
                _registered += 1
            if getattr(self, 'rank', 0) == 0:
                import logging as _lg_v2
                _lg_v2.getLogger('tidalcache').info(
                    '[HOST-COMPRESS] registered %d compress views with TidalCache '
                    '(subsequent ensure_host_allocated will reuse them)',
                    _registered,
                )

'''


# MR_PATCH3: Phase B3 v2 — post-process kv_cache_raw_tensors to replace
# DSA compress group's Device allocation with a Host-hugepage-backed
# NPU-mapped tensor. Achieves real HBM savings while keeping vllm's downstream
# reshape/binding logic unchanged (same shape/dtype from vllm's perspective).
# Gated by TIDALCACHE_HOST_COMPRESS=1.
MR_PATCH3_CODE = '''
        # ── TidalCache Phase B3 v2: replace compress raw_tensor with Host-backed ──
        import os as _tcos_h
        if _tcos_h.environ.get('TIDALCACHE_HOST_COMPRESS', '0') == '1':
            import torch as _torch_h
            import logging as _tclg_h
            _hlog = _tclg_h.getLogger('tidalcache')
            _rank0_h = getattr(self, 'rank', 0) == 0
            try:
                import tidalcache as _tc_h
                _mgr = _tc_h._GLOBAL_MANAGER
            except Exception as _e:
                _mgr = None
                _hlog.warning('[HOST-COMPRESS] TidalCache manager not available: %s', _e)
            if _mgr is not None:
                # Snapshot HBM BEFORE replacement — for drop-verification.
                # Use self.device explicitly (not current_device) so numbers
                # match the worker's actual NPU across DP/TP ranks.
                _hbm_dev = getattr(self, 'device', None)
                try:
                    _hbm_before_alloc = _torch_h.npu.memory_allocated(_hbm_dev) / 1024**3
                    _hbm_before_resv = _torch_h.npu.memory_reserved(_hbm_dev) / 1024**3
                    _hbm_curdev = _torch_h.npu.current_device()
                except Exception:
                    _hbm_before_alloc = _hbm_before_resv = -1.0
                    _hbm_curdev = -1
                # Sanity: does memory_allocated track fresh torch.zeros?
                # If YES → API works, and 0 drop means real refs are held.
                # If NO  → API blind to some allocs; conclusion inverts.
                try:
                    _probe_alloc_before = _torch_h.npu.memory_allocated(_hbm_dev)
                    _probe_t = _torch_h.zeros(1024*1024*256, dtype=_torch_h.int8, device=_hbm_dev)  # 256 MB
                    _probe_alloc_mid = _torch_h.npu.memory_allocated(_hbm_dev)
                    del _probe_t
                    _torch_h.npu.empty_cache()
                    _probe_alloc_after = _torch_h.npu.memory_allocated(_hbm_dev)
                    _hlog.info(
                        '[HOST-COMPRESS] api-probe: allocated %.2f → %.2f → %.2f GB '
                        '(delta up=%.2f MB, delta down=%.2f MB) — expected 256 MB',
                        _probe_alloc_before / 1024**3,
                        _probe_alloc_mid / 1024**3,
                        _probe_alloc_after / 1024**3,
                        (_probe_alloc_mid - _probe_alloc_before) / 1024**2,
                        (_probe_alloc_mid - _probe_alloc_after) / 1024**2,
                    )
                except Exception as _pe:
                    _hlog.warning('[HOST-COMPRESS] api-probe failed: %s', _pe)
                # Detect compress layers by name suffix: '.self_attn.attn'
                _COMPRESS_SFX = '.self_attn.attn'
                # Group by identity: multiple DSA layers may share ONE raw_tensor
                # object (line 4239/4251 branches assign same tensor to all
                # shared_by layers). Replace once per unique tensor object.
                _seen = {}  # id(old) -> new_host_tensor
                _replaced_layers = 0
                _total_bytes = 0
                # Save ONE probe old-tensor so we can inspect its referrers
                # after the loop finishes and after gc.collect runs.
                _probe_old = None
                for _ln, _rt in list(kv_cache_raw_tensors.items()):
                    if not _ln.endswith(_COMPRESS_SFX):
                        continue
                    if _rt is None:
                        continue
                    # raw_tensors entries can be a single tensor OR a tuple
                    # (k_tensor, v_tensor, dsa_k_tensor, ...). For the compress
                    # group in DSV4, we saw single-tensor layout — but handle
                    # both defensively. In tuple case, we replace element [0]
                    # (k_tensor) which is the compress cache buffer.
                    if isinstance(_rt, _torch_h.Tensor):
                        _old_t = _rt
                        _target_slot = 'single'
                    elif isinstance(_rt, (tuple, list)) and len(_rt) > 0 and isinstance(_rt[0], _torch_h.Tensor):
                        _old_t = _rt[0]
                        _target_slot = 'tuple_0'
                    else:
                        continue
                    _key = id(_old_t)
                    if _key in _seen:
                        _new_t = _seen[_key]
                    else:
                        _size = _old_t.numel() * _old_t.element_size()
                        try:
                            _new_t_int8 = _mgr.alloc_group_host_tensor(
                                _size, group_name=f'compress_{_key:x}'
                            )
                            # Reinterpret as the original dtype/shape so
                            # downstream reshape sees an equivalent tensor.
                            _new_t = _new_t_int8.view(_old_t.dtype)[:_old_t.numel()].view(_old_t.shape) \
                                if _old_t.dtype != _torch_h.int8 else _new_t_int8.view(_old_t.shape)
                            _seen[_key] = _new_t
                            _total_bytes += _size
                        except Exception as _e:
                            _hlog.error('[HOST-COMPRESS] failed to alloc for %s: %s', _ln, _e)
                            continue
                    # Stash the very first old tensor for post-loop referrer
                    # analysis (only if we haven't stashed one yet).
                    if _probe_old is None and _rank0_h:
                        _probe_old = _old_t
                    if _target_slot == 'single':
                        kv_cache_raw_tensors[_ln] = _new_t
                    else:
                        _new_tuple = (_new_t,) + tuple(_rt[1:])
                        kv_cache_raw_tensors[_ln] = _new_tuple
                    _replaced_layers += 1
                    if _rank0_h and _replaced_layers <= 2:
                        # Count refs to old tensor BEFORE replacement. Expected
                        # refs (baseline):
                        #   - _rt (loop var)             = 1
                        #   - list(dict.items()) tuple   = 1
                        #   - kv_cache_raw_tensors[_ln]  = 1
                        #   - sys.getrefcount arg        = 1
                        # Total baseline = 4. Anything above = mystery holder.
                        try:
                            import sys as _tcsys
                            _refs = _tcsys.getrefcount(_old_t)
                            _mystery = max(0, _refs - 4)
                        except Exception:
                            _refs = -1
                            _mystery = -1
                        _hlog.info(
                            '[HOST-COMPRESS] %s: replaced Device compress → Host-mapped NPU tensor '
                            '(shape=%s, dtype=%s, %.1f MB, old_dev=%s old_ptr=0x%x '
                            'new_dev=%s new_ptr=0x%x, refcount=%d, mystery_holders≈%d)',
                            _ln, tuple(_old_t.shape), _old_t.dtype,
                            _old_t.numel() * _old_t.element_size() / 1024**2,
                            _old_t.device, _old_t.data_ptr(),
                            _new_t.device, _new_t.data_ptr(),
                            _refs, _mystery,
                        )
                # Drop references to old tensors and force NPU allocator to reclaim.
                _seen.clear()
                try:
                    del _rt
                except Exception:
                    pass
                try:
                    del _old_t
                except Exception:
                    pass
                # Ref-count probe on ONE surviving old tensor — tells us if
                # something else in the enclosing frame is holding a ref.
                # Grab the FIRST replaced entry's *previous* tensor by peeking
                # at kv_cache_raw_tensors after replacement (need a probe target).
                try:
                    import sys as _tcsys
                    # Pick the last _seen key's new tensor's shape to sanity-check;
                    # this doesn't tell us about the OLD tensor, but forces us to
                    # log what's still referenced.
                    _hlog.info(
                        '[HOST-COMPRESS] refprobe: dict size=%d, _seen size=%d, '
                        'first_new_key=%s',
                        len(kv_cache_raw_tensors), len(_seen),
                        next(iter(kv_cache_raw_tensors), 'N/A'),
                    )
                except Exception:
                    pass
                # Force Python GC BEFORE empty_cache — dict replace may leave
                # transient refs from `list(dict.items())` or the loop frame.
                try:
                    import gc as _tcgc
                    _n_before_gc = _tcgc.collect()
                    _hlog.info('[HOST-COMPRESS] gc.collect returned %d', _n_before_gc)
                except Exception:
                    pass
                # Deep probe: enumerate referrers of ONE stashed old tensor.
                # At this point locals _rt/_old_t are deleted and gc ran.
                # Baseline expected refs: _probe_old local var (1) + get_referrers
                # arg (transient, doesn't show). Anything else is a MYSTERY holder.
                try:
                    if _rank0_h and _probe_old is not None:
                        import sys as _tcsys
                        import gc as _tcgc2
                        _pr_refs = _tcsys.getrefcount(_probe_old)
                        _pr_referrers = _tcgc2.get_referrers(_probe_old)
                        _hlog.info(
                            '[HOST-COMPRESS] probe refcount=%d, referrers_count=%d',
                            _pr_refs, len(_pr_referrers),
                        )
                        for _i, _r in enumerate(_pr_referrers[:5]):
                            _t = type(_r).__name__
                            _rid = id(_r)
                            _summary = ''
                            if isinstance(_r, dict):
                                _summary = f'dict len={len(_r)} sample_keys={list(_r.keys())[:3]}'
                            elif isinstance(_r, (list, tuple)):
                                _summary = f'{_t} len={len(_r)}'
                            elif hasattr(_r, 'f_code'):  # frame
                                _summary = f'frame func={_r.f_code.co_name} file={_r.f_code.co_filename}:{_r.f_lineno}'
                            _hlog.info(
                                '[HOST-COMPRESS] referrer[%d]: type=%s id=0x%x %s',
                                _i, _t, _rid, _summary,
                            )
                            # If it's kv_cache_raw_tensors, list ALL keys whose
                            # value still points at the old tensor.
                            if isinstance(_r, dict) and _rid == id(kv_cache_raw_tensors):
                                _stubborn_keys = []
                                for _k, _v in _r.items():
                                    if _v is _probe_old:
                                        _stubborn_keys.append(_k)
                                    elif isinstance(_v, (tuple, list)):
                                        for _idx, _elem in enumerate(_v):
                                            if _elem is _probe_old:
                                                _stubborn_keys.append(f'{_k}[{_idx}]')
                                _hlog.info(
                                    '[HOST-COMPRESS] stubborn refs in kv_cache_raw_tensors: %s',
                                    _stubborn_keys,
                                )
                        del _pr_referrers
                except Exception as _pe:
                    _hlog.warning('[HOST-COMPRESS] referrer probe failed: %s', _pe)
                # Release the probe ref
                _probe_old = None
                # HBM AFTER dict replace + gc, BEFORE empty_cache
                try:
                    _hbm_mid_alloc = _torch_h.npu.memory_allocated(_hbm_dev) / 1024**3
                except Exception:
                    _hbm_mid_alloc = -1.0
                try:
                    _torch_h.npu.empty_cache()
                except Exception:
                    pass
                # HBM AFTER empty_cache — shows what allocator returned to driver
                try:
                    _hbm_after_alloc = _torch_h.npu.memory_allocated(_hbm_dev) / 1024**3
                    _hbm_after_resv = _torch_h.npu.memory_reserved(_hbm_dev) / 1024**3
                except Exception:
                    _hbm_after_alloc = _hbm_after_resv = -1.0
                _hlog.info(
                    '[HOST-COMPRESS] done: replaced=%d layers, total=%.2f GB → Host '
                    '(rank=%d)',
                    _replaced_layers, _total_bytes / 1024**3,
                    getattr(self, 'rank', 0),
                )
                _hlog.info(
                    '[HOST-COMPRESS] HBM diag: allocated %.2f→%.2f→%.2f GB, '
                    'reserved %.2f→%.2f GB (expected drop ~%.2f GB) '
                    '[dev=%s curdev=%d]',
                    _hbm_before_alloc, _hbm_mid_alloc, _hbm_after_alloc,
                    _hbm_before_resv, _hbm_after_resv,
                    _total_bytes / 1024**3,
                    _hbm_dev, _hbm_curdev,
                )

'''


def find_and_patch_mr_init_kv(content):
    """Insert:
      - MR_PATCH2_INIT before `kv_cache_raw_tensors = self._allocate_kv_cache_tensors(...)`
        (so manager exists when MR_PATCH3 fires inside _allocate_kv_cache_tensors)
      - MR_PATCH2 (HBM log) before `return kv_caches`
      - MR_PATCH3 before `return kv_cache_raw_tensors`
    """
    # PATCH2_INIT: before the allocate call inside initialize_kv_cache_tensors
    init_func = re.search(r'def initialize_kv_cache_tensors\(self.*?\n', content)
    if not init_func:
        return None
    init_call_pat = re.compile(
        r'^(        kv_cache_raw_tensors = self\._allocate_kv_cache_tensors\(kv_cache_config\))',
        re.MULTILINE,
    )
    init_m = init_call_pat.search(content, init_func.start())
    if init_m:
        content = content[:init_m.start()] + MR_PATCH2_INIT_CODE + content[init_m.start():]
    else:
        print("  WARNING: MR_PATCH2_INIT anchor not found — skipping early manager init")

    # PATCH2: HBM log — before 'return kv_caches'
    ret2 = re.compile(r'^        return kv_caches\s*$', re.MULTILINE)
    m2 = ret2.search(content, init_func.start())
    if not m2:
        return None
    content = content[:m2.start()] + MR_PATCH2_CODE + content[m2.start():]

    # PATCH3: _allocate_kv_cache_tensors → before 'return kv_cache_raw_tensors'
    func3 = re.search(r'def _allocate_kv_cache_tensors\(self.*?\n', content)
    if not func3:
        print("  WARNING: MR_PATCH3 anchor (_allocate_kv_cache_tensors) not found — skipping")
        return content
    ret3 = re.compile(r'^        return kv_cache_raw_tensors\s*$', re.MULTILINE)
    m3 = ret3.search(content, func3.start())
    if not m3:
        print("  WARNING: MR_PATCH3 'return kv_cache_raw_tensors' not found — skipping")
        return content
    content = content[:m3.start()] + MR_PATCH3_CODE + content[m3.start():]
    return content


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    vllm_dir = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "--apply"

    dsa_path = os.path.join(vllm_dir, "vllm_ascend/attention/dsa_v1.py")
    mr_path = os.path.join(vllm_dir, "vllm_ascend/worker/model_runner_v1.py")
    dsa_cp_path = os.path.join(vllm_dir, "vllm_ascend/attention/context_parallel/dsa_cp.py")

    for p in [dsa_path, mr_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)
    has_dsa_cp = os.path.exists(dsa_cp_path)

    if action == "--rollback":
        print("=== TidalCache Rollback ===")
        for p in [dsa_path, mr_path, dsa_cp_path]:
            bak = p + ".bak"
            if os.path.exists(bak):
                shutil.copy2(bak, p)
                print(f"  Restored: {p}")
            else:
                print(f"  No backup: {bak}")
        print("Done.")
        return

    dry_run = action == "--check"
    print(f"=== TidalCache Patch {'(dry-run)' if dry_run else ''} ===")

    # Patch dsa_v1.py
    print(f"\n--- {dsa_path} ---")
    with open(dsa_path, "r") as f:
        dsa_content = f.read()

    if MARKER in dsa_content:
        print("  SKIP (already patched)")
    else:
        # PATCH1, PATCH3, PATCH4 via string matching
        for name, (anchor, insertion, mode) in DSA_PATCHES.items():
            if anchor not in dsa_content:
                print(f"  ERROR: anchor not found for {name}")
                print(f"    Expected: {anchor[:80]}...")
                sys.exit(1)
            if mode == "after":
                dsa_content = dsa_content.replace(anchor, anchor + insertion, 1)
            elif mode == "replace":
                dsa_content = dsa_content.replace(anchor, insertion, 1)
            print(f"  {name}: OK")

        # PATCH2: regex — insert gather code after _update_indexcache_topk_indices
        # in the decode path, before attn_op. Handle extra lines between them.
        p2_pattern = re.compile(
            r'(            if self\.compress_ratio == 4 and self\.use_index_cache:\n'
            r'                self\._update_indexcache_topk_indices\(compress_topk_idxs, offset=0\)\n)'
            r'(\n(?:        [^\n]*\n)*?)'  # any lines between (same or lower indent)
            r'(        attn_op = DeviceOperator\.get_dsa_sparse_attn_op\(\))'
        )
        m = p2_pattern.search(dsa_content)
        if m is None:
            print("  ERROR: PATCH2_gather regex not matched")
            sys.exit(1)
        dsa_content = (
            dsa_content[:m.start(3)]
            + PATCH2_GATHER_CODE
            + m.group(3)
            + dsa_content[m.end(3):]
        )
        print("  PATCH2_gather: OK")

        # PATCH3: regex — patch ALL scatter calls (prefill + decode) with
        # mode-aware branches. Iterates all matches from end to start so that
        # position offsets stay valid during replacement.
        #   TIDALCACHE_PREFILL_MODE=B: dual write to Device compress_kv_cache
        #                              AND Host hugepage (npu view). Host gets
        #                              full KV data for both prefill and decode.
        #   TIDALCACHE_PREFILL_MODE=A (or unset): single write to Device only.
        #                              Post-prefill D2H sweep (TBD) will move
        #                              data to Host after prefill completes.
        #   TIDALCACHE_PREFILL_MODE=OFF: bypass all offload writes (compat).
        p3_pattern = re.compile(
            r'( +)(DeviceOperator\.dsa_kv_compress_scatter\()'
            r'\s*compress_kv_cache,\s*(.*?\))',
            re.DOTALL,
        )
        matches = list(p3_pattern.finditer(dsa_content))
        if not matches:
            print("  ERROR: PATCH3_scatter regex not matched")
            sys.exit(1)
        scatter = "DeviceOperator.dsa_kv_compress_scatter"
        for m3 in reversed(matches):  # reversed to keep earlier positions stable
            indent = m3.group(1)
            args_inner = m3.group(3).rstrip(")").strip()
            replacement = (
                f"{indent}# ── TidalCache: mode-aware scatter (A=Device only, B=dual) ──\n"
                f"{indent}import os as _tc_os_s\n"
                f"{indent}_tc_mode_s = _tc_os_s.environ.get('TIDALCACHE_PREFILL_MODE', 'A')\n"
                f"{indent}if self.kv_offload_enabled and _tc_mode_s != 'OFF':\n"
                f"{indent}    if self._tidalcache_mgr is None:\n"
                f"{indent}        import tidalcache as _tc_s\n"
                f"{indent}        self._tidalcache_mgr = _tc_s._GLOBAL_MANAGER\n"
                f"{indent}    if self._tidalcache_mgr is not None and layer_name not in self._tidalcache_mgr.layers:\n"
                f"{indent}        self._tidalcache_mgr.ensure_host_allocated(layer_name)\n"
                f"{indent}    {scatter}(compress_kv_cache, {args_inner})\n"
                f"{indent}    if _tc_mode_s == 'B' and self._tidalcache_mgr is not None and layer_name in self._tidalcache_mgr.layers:\n"
                f"{indent}        _host_kv = self._tidalcache_mgr.layers[layer_name].npu_kv_cache\n"
                f"{indent}        {scatter}(_host_kv, {args_inner})\n"
                f"{indent}        import logging as _lg_s\n"
                f"{indent}        _lg_s_ = _lg_s.getLogger('tidalcache')\n"
                f"{indent}        if not getattr(self, '_tc_logged_mode_' + layer_name.replace('.','_'), False):\n"
                f"{indent}            _lg_s_.info('[SCATTER-B first] %s → dual write (device+host)', layer_name)\n"
                f"{indent}            setattr(self, '_tc_logged_mode_' + layer_name.replace('.','_'), True)\n"
                f"{indent}        _lg_s_.debug('[SCATTER-B] %s → device+host', layer_name)\n"
                f"{indent}    else:\n"
                f"{indent}        import logging as _lg_s\n"
                f"{indent}        _lg_s_ = _lg_s.getLogger('tidalcache')\n"
                f"{indent}        if not getattr(self, '_tc_logged_mode_' + layer_name.replace('.','_'), False):\n"
                f"{indent}            _lg_s_.info('[SCATTER-A first] %s → device only', layer_name)\n"
                f"{indent}            setattr(self, '_tc_logged_mode_' + layer_name.replace('.','_'), True)\n"
                f"{indent}        _lg_s_.debug('[SCATTER-A] %s → device only', layer_name)\n"
                f"{indent}else:\n"
                f"{indent}    {scatter}(compress_kv_cache, {args_inner})"
            )
            dsa_content = (
                dsa_content[:m3.start()]
                + replacement
                + dsa_content[m3.end():]
            )
        print(f"  PATCH3_scatter: OK ({len(matches)} call sites patched)")

        # PATCH4: rewrite attn_op cmp_block_table arg to use _tc_cmp_block_table.
        # PATCH2 sets _tc_cmp_block_table on every decode entry (fallback to the
        # original block_table when TidalCache is off). When Phase B2 is active
        # (TIDALCACHE_ATTN_ON_SEL=1) it points at the small mini_cmp_block_table
        # so attn_op reads from sel_kv directly. Only rewrite in the decode path.
        p4_pattern = re.compile(
            r'cmp_block_table=compressor_decode_metadata\.block_table,'
        )
        p4_matches = list(p4_pattern.finditer(dsa_content))
        if not p4_matches:
            print("  WARNING: PATCH4_attn_arg no matches (decode attn_op cmp_block_table)")
        else:
            dsa_content = p4_pattern.sub(
                'cmp_block_table=_tc_cmp_block_table,',
                dsa_content,
            )
            print(f"  PATCH4_attn_arg: OK ({len(p4_matches)} call sites rewritten)")

        if not dry_run:
            bak = dsa_path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(dsa_path, bak)
                print(f"  Backed up: {bak}")
            with open(dsa_path, "w") as f:
                f.write(dsa_content)
            print(f"  Written: {dsa_path}")
        else:
            print("  (dry-run, not written)")

    # Patch model_runner_v1.py
    print(f"\n--- {mr_path} ---")
    with open(mr_path, "r") as f:
        mr_content = f.read()

    if MARKER in mr_content:
        print("  SKIP (already patched)")
    else:
        # PATCH1: add flag
        if MR_PATCH1_ANCHOR not in mr_content:
            print("  ERROR: PATCH1 anchor not found")
            sys.exit(1)
        mr_content = mr_content.replace(
            MR_PATCH1_ANCHOR,
            MR_PATCH1_INSERT + MR_PATCH1_ANCHOR,
            1,
        )
        print("  PATCH1_init: OK")

        # PATCH2: init TidalCache
        result = find_and_patch_mr_init_kv(mr_content)
        if result is None:
            print("  ERROR: PATCH2 anchor not found")
            sys.exit(1)
        mr_content = result
        print("  PATCH2_init_kv: OK")

        if not dry_run:
            bak = mr_path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(mr_path, bak)
                print(f"  Backed up: {bak}")
            with open(mr_path, "w") as f:
                f.write(mr_content)
            print(f"  Written: {mr_path}")
        else:
            print("  (dry-run, not written)")

    # Patch dsa_cp.py (V4 Context-Parallel path)
    if has_dsa_cp:
        print(f"\n--- {dsa_cp_path} ---")
        with open(dsa_cp_path, "r") as f:
            cp_content = f.read()

        if MARKER in cp_content:
            print("  SKIP (already patched)")
        else:
            # CP_PATCH1: init — add kv_offload attrs after index_topk assignment
            cp1_anchor = "self.index_topk = self.indexer.index_topk"
            if cp1_anchor not in cp_content:
                print("  ERROR: CP_PATCH1 anchor not found")
                sys.exit(1)
            cp1_insert = cp1_anchor + '''

            # ── TidalCache: KV offload ──
            from tidalcache import TIDALCACHE_ENABLED
            self.kv_offload_enabled = TIDALCACHE_ENABLED
            self._tidalcache_mgr = None'''
            cp_content = cp_content.replace(cp1_anchor, cp1_insert, 1)
            print("  CP_PATCH1_init: OK")

            # CP_PATCH2: gather — insert before attn_op in _forward
            # Use the same anchor: attn_op = DeviceOperator.get_dsa_sparse_attn_op()
            # But scope it to _forward by requiring notify_kv_cache_written before it
            cp2_pattern = re.compile(
                r'(        notify_kv_cache_written\(layer_name\)\n'
                r'        record_attention_compute_start\(\)\n)'
                r'(        attn_op = DeviceOperator\.get_dsa_sparse_attn_op\(\))'
            )
            m_cp2 = cp2_pattern.search(cp_content)
            if m_cp2 is None:
                print("  ERROR: CP_PATCH2 gather anchor not found")
                sys.exit(1)
            cp2_gather = '''
        # ── TidalCache: lazy init + Sparse Host→Device Gather (CP) ──
        # compressor_attn_metadata only assigned in compress_ratio==4/128 branch;
        # use locals().get() so this code is a no-op for other compress_ratios.
        _cam = locals().get('compressor_attn_metadata')
        _tc_cmp_block_table = _cam.req_metadata.block_table if _cam is not None else None
        # Save reference for compress-poison test (Step 1 of B3)
        _orig_compress_ref = compress_kv_cache
        if getattr(self, 'kv_offload_enabled', False) and _cam is not None:
            if self._tidalcache_mgr is None:
                import tidalcache as _tc
                self._tidalcache_mgr = _tc._GLOBAL_MANAGER
            if self._tidalcache_mgr is not None:
                import torch as _torch
                import logging as _logging
                import os as _tcos_b2cp
                _tclog = _logging.getLogger("tidalcache")
                _B = hidden_states.shape[0]
                _local_topk = compress_topk_idxs.numel() // _B
                if layer_name not in self._tidalcache_mgr.layers:
                    self._tidalcache_mgr.alloc_layer(layer_name, local_topk=_local_topk)
                _sel_kv, _sel_rope, _sel_actual_seq = self._tidalcache_mgr.gather(
                    layer_name=layer_name,
                    topk_indices=compress_topk_idxs.view(_B, 1, 1, _local_topk),
                    full_block_table=compressor_attn_metadata.req_metadata.block_table,
                    full_actual_seq=local_seq_lengths_key,
                    full_q_actual_seq=_torch.ones(_B, dtype=_torch.int32, device=hidden_states.device),
                )
                if _tcos_b2cp.environ.get("TIDALCACHE_POISON", "0") == "1":
                    _sel_kv.fill_(-1000.0)
                    _sel_rope.fill_(-1000.0)
                    _tclog.info("[POISON-CP] %s gather output filled with -1000", layer_name)

                _state = self._tidalcache_mgr.layers[layer_name]
                _attn_on_sel = _tcos_b2cp.environ.get("TIDALCACHE_ATTN_ON_SEL", "0") == "1"
                _max_batch = _state.mini_cmp_block_table.shape[0]
                # attn_op's query B is num REQUESTS = original block_table dim 0
                _actual_reqs = _cam.req_metadata.block_table.shape[0]
                if _attn_on_sel and _state.sel_kv_cache is not None and _actual_reqs <= _max_batch:
                    # Phase B2 (CP): attn_op reads sel-side directly, no copy-back.
                    compress_kv_cache = _state.mini_compress_kv
                    _orig_shape = tuple(compress_topk_idxs.shape)
                    _topk_actual = _orig_shape[-1]
                    _flat_idx = _torch.arange(
                        _topk_actual,
                        dtype=compress_topk_idxs.dtype,
                        device=hidden_states.device,
                    )
                    _bcast_shape = [1] * (len(_orig_shape) - 1) + [_topk_actual]
                    compress_topk_idxs = _flat_idx.view(*_bcast_shape).expand(*_orig_shape).contiguous()
                    _tc_cmp_block_table = _state.mini_cmp_block_table[:_actual_reqs]
                    if not getattr(self, '_tc_logged_b2cp_' + layer_name.replace('.','_'), False):
                        _tclog.info(
                            '[ATTN-ON-SEL-CP first] %s → attn reads sel_kv (Phase B2 CP), reqs=%d, orig_shape=%s',
                            layer_name, _actual_reqs, _orig_shape,
                        )
                        setattr(self, '_tc_logged_b2cp_' + layer_name.replace('.','_'), True)
                    _tclog.debug("[ATTN-ON-SEL-CP] %s B=%d reqs=%d", layer_name, _B, _actual_reqs)
                else:
                    # Phase B1: copy-back to compress_kv_cache.
                    _cbs = self._tidalcache_mgr.compress_block_size
                    _gpb = compress_kv_cache.shape[1] // _cbs
                    _bt_full = compressor_attn_metadata.req_metadata.block_table
                    _aB = min(_B, _bt_full.shape[0])
                    _tidx = compress_topk_idxs.view(_B, _local_topk)[:_aB]
                    _bseq = (_tidx // _gpb).long()
                    _goff = (_tidx % _gpb).long()
                    _cbt = _bt_full[:_aB].long()
                    _pblk = _torch.gather(_cbt, 1, _bseq)
                    _dst = (_pblk * _gpb + _goff).view(-1)
                    _n = _aB * _local_topk
                    _trail = compress_kv_cache.shape[2:]
                    _cmp64 = compress_kv_cache.view(-1, _cbs, *_trail)
                    _src = _sel_kv[:_n]
                    if _src.shape[2:] != _trail:
                        _src = _src.view(_n, _cbs, *_trail)
                    _cmp64[_dst] = _src
                    _tclog.debug("[COPYBACK-CP] %s B=%d aB=%d topk=%d dst_blocks=%d", layer_name, _B, _aB, _local_topk, _n)

                # ── Step 1 test (CP): poison vllm's compress_kv_cache ──
                if _tcos_b2cp.environ.get("TIDALCACHE_POISON_COMPRESS", "0") == "1":
                    _orig_compress_ref.zero_()
                    if not getattr(self, '_tc_poison_c_cp_' + layer_name.replace('.','_'), False):
                        _tclog.info(
                            '[POISON-COMPRESS-CP first] %s → vllm compress_kv_cache zeroed',
                            layer_name,
                        )
                        setattr(self, '_tc_poison_c_cp_' + layer_name.replace('.','_'), True)

'''
            cp_content = (
                cp_content[:m_cp2.end(1)]
                + cp2_gather
                + m_cp2.group(2)
                + cp_content[m_cp2.end(2):]
            )
            print("  CP_PATCH2_gather: OK")

            # CP_PATCH3: patch ALL scatter calls with mode-aware branches
            #   TIDALCACHE_PREFILL_MODE=B: dual write (Device + Host)
            #   TIDALCACHE_PREFILL_MODE=A/unset: Device only (sweep TBD)
            #   TIDALCACHE_PREFILL_MODE=OFF: bypass
            cp3_pattern = re.compile(
                r'( +)(DeviceOperator\.dsa_kv_compress_scatter\()'
                r'\s*compress_kv_cache,\s*(.*?\))',
                re.DOTALL,
            )
            cp3_matches = list(cp3_pattern.finditer(cp_content))
            if not cp3_matches:
                print("  ERROR: CP_PATCH3 scatter not found")
                sys.exit(1)
            scatter = "DeviceOperator.dsa_kv_compress_scatter"
            for m_cp3 in reversed(cp3_matches):
                cp3_indent = m_cp3.group(1)
                cp3_args = m_cp3.group(3).rstrip(")").strip()
                cp3_replace = (
                    f"{cp3_indent}# ── TidalCache: mode-aware scatter (CP) ──\n"
                    f"{cp3_indent}import os as _tc_os_cp\n"
                    f"{cp3_indent}_tc_mode_cp = _tc_os_cp.environ.get('TIDALCACHE_PREFILL_MODE', 'A')\n"
                    f"{cp3_indent}if getattr(self, 'kv_offload_enabled', False) and _tc_mode_cp != 'OFF':\n"
                    f"{cp3_indent}    if self._tidalcache_mgr is None:\n"
                    f"{cp3_indent}        import tidalcache as _tc_cp\n"
                    f"{cp3_indent}        self._tidalcache_mgr = _tc_cp._GLOBAL_MANAGER\n"
                    f"{cp3_indent}    if self._tidalcache_mgr is not None and layer_name not in self._tidalcache_mgr.layers:\n"
                    f"{cp3_indent}        self._tidalcache_mgr.ensure_host_allocated(layer_name)\n"
                    f"{cp3_indent}    {scatter}(compress_kv_cache, {cp3_args})\n"
                    f"{cp3_indent}    if _tc_mode_cp == 'B' and self._tidalcache_mgr is not None and layer_name in self._tidalcache_mgr.layers:\n"
                    f"{cp3_indent}        _host_kv_cp = self._tidalcache_mgr.layers[layer_name].npu_kv_cache\n"
                    f"{cp3_indent}        {scatter}(_host_kv_cp, {cp3_args})\n"
                    f"{cp3_indent}        import logging as _lg_cp\n"
                    f"{cp3_indent}        _lg_cp_ = _lg_cp.getLogger('tidalcache')\n"
                    f"{cp3_indent}        if not getattr(self, '_tc_logged_mode_cp_' + layer_name.replace('.','_'), False):\n"
                    f"{cp3_indent}            _lg_cp_.info('[SCATTER-B-CP first] %s → dual write (device+host)', layer_name)\n"
                    f"{cp3_indent}            setattr(self, '_tc_logged_mode_cp_' + layer_name.replace('.','_'), True)\n"
                    f"{cp3_indent}        _lg_cp_.debug('[SCATTER-B-CP] %s → device+host', layer_name)\n"
                    f"{cp3_indent}    else:\n"
                    f"{cp3_indent}        import logging as _lg_cp\n"
                    f"{cp3_indent}        _lg_cp_ = _lg_cp.getLogger('tidalcache')\n"
                    f"{cp3_indent}        if not getattr(self, '_tc_logged_mode_cp_' + layer_name.replace('.','_'), False):\n"
                    f"{cp3_indent}            _lg_cp_.info('[SCATTER-A-CP first] %s → device only', layer_name)\n"
                    f"{cp3_indent}            setattr(self, '_tc_logged_mode_cp_' + layer_name.replace('.','_'), True)\n"
                    f"{cp3_indent}        _lg_cp_.debug('[SCATTER-A-CP] %s → device only', layer_name)\n"
                    f"{cp3_indent}else:\n"
                    f"{cp3_indent}    {scatter}(compress_kv_cache, {cp3_args})"
                )
                cp_content = (
                    cp_content[:m_cp3.start()]
                    + cp3_replace
                    + cp_content[m_cp3.end():]
                )
            print(f"  CP_PATCH3_scatter: OK ({len(cp3_matches)} call sites patched)")

            # CP_PATCH4: rewrite attn_op cmp_block_table to _tc_cmp_block_table
            # CP_PATCH2 sets _tc_cmp_block_table on every decode entry (fallback
            # to compressor_attn_metadata.req_metadata.block_table when TidalCache
            # is off). Phase B2 (TIDALCACHE_ATTN_ON_SEL=1) points it at the small
            # mini_cmp_block_table so attn_op reads from sel_kv directly.
            cp4_pattern = re.compile(
                r'cmp_block_table=compressor_attn_metadata\.req_metadata\.block_table,'
            )
            cp4_matches = list(cp4_pattern.finditer(cp_content))
            if not cp4_matches:
                print("  WARNING: CP_PATCH4 no matches")
            else:
                cp_content = cp4_pattern.sub(
                    'cmp_block_table=_tc_cmp_block_table,',
                    cp_content,
                )
                print(f"  CP_PATCH4_attn_arg: OK ({len(cp4_matches)} call sites rewritten)")

            if not dry_run:
                bak = dsa_cp_path + ".bak"
                if not os.path.exists(bak):
                    shutil.copy2(dsa_cp_path, bak)
                    print(f"  Backed up: {bak}")
                with open(dsa_cp_path, "w") as f:
                    f.write(cp_content)
                print(f"  Written: {dsa_cp_path}")
            else:
                print("  (dry-run, not written)")
    else:
        print(f"\n--- dsa_cp.py not found (V3-only mode) ---")

    print("\n=== Done ===")
    if not dry_run:
        print("Enable with: export VLLM_DSA_KV_OFFLOAD=1")
        print("Rollback:    python3 apply_patches.py", vllm_dir, "--rollback")


if __name__ == "__main__":
    main()
