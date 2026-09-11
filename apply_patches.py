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
        if self.kv_offload_enabled:
            if self._tidalcache_mgr is None:
                import tidalcache as _tc
                self._tidalcache_mgr = _tc._GLOBAL_MANAGER
            if self._tidalcache_mgr is not None:
                import torch as _torch
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
                # Copy gathered groups into compress_kv_cache at original positions.
                # This preserves the 4D shape and block_size=128 that attn_op expects.
                _cbs = self._tidalcache_mgr.compress_block_size  # 64
                _gpb = compress_kv_cache.shape[1] // _cbs  # groups per block (128/64=2)
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

# PATCH2: initialize TidalCache in initialize_kv_cache_tensors
# Anchor: "return kv_caches" at the end of initialize_kv_cache_tensors
# We need a unique anchor — use the function's return + its next method def
MR_PATCH2_CODE = '''
        # ── TidalCache: create manager, layers allocated lazily in dsa_v1 ──
        _hf_cfg = getattr(self.model_config, 'hf_text_config', None)
        _has_topk = _hf_cfg is not None and hasattr(_hf_cfg, 'index_topk')
        if self.kv_offload_enabled and _has_topk:
            import torch as _torch
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

'''


def find_and_patch_mr_init_kv(content):
    """Find the return statement in initialize_kv_cache_tensors and insert before it."""
    # Find the function
    func_match = re.search(
        r'def initialize_kv_cache_tensors\(self.*?\n',
        content
    )
    if not func_match:
        return None

    func_start = func_match.start()

    # Find "return kv_caches" after the function start
    # Look for "        return kv_caches\n" (8-space indent = method body)
    return_pattern = re.compile(r'^        return kv_caches\s*$', re.MULTILINE)
    match = return_pattern.search(content, func_start)
    if not match:
        return None

    # Insert before the return
    return content[:match.start()] + MR_PATCH2_CODE + content[match.start():]


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

        # PATCH3: regex — replace scatter target with offload-aware branch
        # Match scatter call where first arg is compress_kv_cache (decode path).
        # Capture indent and remaining args (may be single-line or multi-line).
        p3_pattern = re.compile(
            r'( +)(DeviceOperator\.dsa_kv_compress_scatter\()'
            r'\s*compress_kv_cache,\s*(.*?\))',
            re.DOTALL,
        )
        m3 = p3_pattern.search(dsa_content)
        if m3 is None:
            print("  ERROR: PATCH3_scatter regex not matched")
            sys.exit(1)
        indent = m3.group(1)
        # Normalize rest_args: collapse whitespace, extract just the args
        rest_args_raw = m3.group(3)
        # rest_args_raw is like "compressed_kv, compress_slot_mapping)" (with possible whitespace/newlines)
        # Strip the trailing ) and normalize whitespace
        args_inner = rest_args_raw.rstrip(")").strip()
        # args_inner is now like "compressed_kv, compress_slot_mapping"
        scatter = "DeviceOperator.dsa_kv_compress_scatter"
        replacement = (
            f"{indent}# ── TidalCache: scatter to Host NPU view ──\n"
            f"{indent}if self.kv_offload_enabled and self._tidalcache_mgr is not None:\n"
            f"{indent}    host_kv = self._tidalcache_mgr.layers[layer_name].npu_kv_cache\n"
            f"{indent}    {scatter}(host_kv, {args_inner})\n"
            f"{indent}else:\n"
            f"{indent}    {scatter}(compress_kv_cache, {args_inner})"
        )
        dsa_content = (
            dsa_content[:m3.start()]
            + replacement
            + dsa_content[m3.end():]
        )
        print("  PATCH3_scatter: OK")

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
        # ── TidalCache: lazy init + Sparse Host→Device Gather ──
        if getattr(self, 'kv_offload_enabled', False):
            if self._tidalcache_mgr is None:
                import tidalcache as _tc
                self._tidalcache_mgr = _tc._GLOBAL_MANAGER
            if self._tidalcache_mgr is not None:
                import torch as _torch
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
                # Copy gathered groups into compress_kv_cache at original positions.
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

'''
            cp_content = (
                cp_content[:m_cp2.end(1)]
                + cp2_gather
                + m_cp2.group(2)
                + cp_content[m_cp2.end(2):]
            )
            print("  CP_PATCH2_gather: OK")

            # CP_PATCH3: scatter — replace scatter target
            cp3_pattern = re.compile(
                r'( +)(DeviceOperator\.dsa_kv_compress_scatter\()'
                r'\s*compress_kv_cache,\s*(.*?\))',
                re.DOTALL,
            )
            m_cp3 = cp3_pattern.search(cp_content)
            if m_cp3 is None:
                print("  ERROR: CP_PATCH3 scatter not found")
                sys.exit(1)
            cp3_indent = m_cp3.group(1)
            cp3_args = m_cp3.group(3).rstrip(")").strip()
            scatter = "DeviceOperator.dsa_kv_compress_scatter"
            cp3_replace = (
                f"{cp3_indent}# ── TidalCache: scatter to Host NPU view ──\n"
                f"{cp3_indent}if getattr(self, 'kv_offload_enabled', False) and self._tidalcache_mgr is not None:\n"
                f"{cp3_indent}    host_kv = self._tidalcache_mgr.layers[layer_name].npu_kv_cache\n"
                f"{cp3_indent}    {scatter}(host_kv, {cp3_args})\n"
                f"{cp3_indent}else:\n"
                f"{cp3_indent}    {scatter}(compress_kv_cache, {cp3_args})"
            )
            cp_content = (
                cp_content[:m_cp3.start()]
                + cp3_replace
                + cp_content[m_cp3.end():]
            )
            print("  CP_PATCH3_scatter: OK")

            # CP_PATCH4: removed — with copy-back approach, attn_op uses original
            # compress_kv_cache and block_table unchanged.
            print("  CP_PATCH4: SKIPPED (copy-back approach)")

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
