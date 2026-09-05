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

# PATCH2: insert gather after _update_indexcache_topk_indices, before attn_op
# Use regex to handle version differences (some versions have extra lines between)
PATCH2_GATHER_CODE = '''
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
                compress_kv_cache = sel_kv
'''

# PATCH3: scatter redirect — replace dsa_kv_compress_scatter target
# Use regex: find the scatter call with compress_kv_cache as first arg (decode path)
# Capture the remaining args so the replacement preserves them exactly.

# PATCH4: attn_op block_table replacement
# Match the unique decode-path attn_op call
DSA_PATCH4_ANCHOR = '''        elif self.compress_ratio == 4:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=compressor_decode_metadata.block_table,'''

DSA_PATCH4_REPLACE = '''        elif self.compress_ratio == 4:
            # ── TidalCache: use selection block_table ──
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
                cmp_block_table=_cmp_block_table,'''

DSA_PATCHES["PATCH4_attn_op"] = (DSA_PATCH4_ANCHOR, DSA_PATCH4_REPLACE, "replace")


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
        # ── TidalCache: allocate Host KV + Selection Cache ──
        _hf_cfg = getattr(self.model_config, 'hf_text_config', None)
        _has_topk = _hf_cfg is not None and hasattr(_hf_cfg, 'index_topk')
        if self.kv_offload_enabled and _has_topk:
            import torch as _torch
            from tidalcache.offload_manager import TidalCacheManager

            hf_config = _hf_cfg
            index_topk = getattr(hf_config, 'index_topk', 512)
            # V3: kv_lora_rank; V4: head_dim
            kv_dim = getattr(hf_config, 'kv_lora_rank', None)
            if kv_dim is None:
                kv_dim = getattr(hf_config, 'head_dim', 512)
            qk_rope_head_dim = getattr(hf_config, 'qk_rope_head_dim', 64)
            # Get block_size: try hf_config first, then kv_cache_config
            block_size = getattr(hf_config, 'compress_block_size', None)
            if block_size is None:
                try:
                    _grp = kv_cache_config.kv_cache_groups[0]
                    _spec = list(_grp.kv_cache_spec.values())[0]
                    block_size = _spec.block_size
                except (AttributeError, IndexError, KeyError):
                    block_size = self.cache_config.block_size

            # V4 CSA/HCA layer filtering via compress_ratios
            compress_ratios = getattr(hf_config, 'compress_ratios', None)

            kv_dtype = self.model_config.dtype
            rope_dtype = self.model_config.dtype
            if getattr(hf_config, 'kv_cache_fp8', False):
                kv_dtype = _torch.float8_e4m3fn
                rope_dtype = _torch.bfloat16

            self._tidalcache_mgr = TidalCacheManager(
                num_blocks=kv_cache_config.num_blocks,
                block_size=block_size,
                kv_dim=kv_dim,
                rope_dim=qk_rope_head_dim,
                index_topk=index_topk,
                max_batch_size=self.scheduler_config.max_num_seqs,
                dtype=kv_dtype,
                device=self.device,
                rope_dtype=rope_dtype,
            )

            num_layers_allocated = 0
            for layer_idx, layer_name in enumerate(kv_caches):
                # Skip HCA layers (compress_ratio=128) and dense layers (0)
                if compress_ratios is not None:
                    if layer_idx < len(compress_ratios) and compress_ratios[
                            layer_idx] not in (4,):
                        continue

                ctx = self.compilation_config.static_forward_context.get(
                    layer_name)
                if ctx is None:
                    continue
                dsa_attn = getattr(ctx, 'dsa_attn', None)
                if dsa_attn is None:
                    continue
                impl = getattr(dsa_attn, 'impl', None)
                if impl is None or not getattr(
                        impl, 'kv_offload_enabled', False):
                    continue

                self._tidalcache_mgr.alloc_layer(layer_name)
                impl._tidalcache_mgr = self._tidalcache_mgr
                num_layers_allocated += 1

            logger.info(
                "TidalCache: initialized %d layers, topk=%d, blocks=%d, "
                "kv_dim=%d, block_size=%d",
                num_layers_allocated, index_topk,
                kv_cache_config.num_blocks, kv_dim, block_size,
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

    for p in [dsa_path, mr_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)

    if action == "--rollback":
        print("=== TidalCache Rollback ===")
        for p in [dsa_path, mr_path]:
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
            dsa_content[:m.end(1)]
            + PATCH2_GATHER_CODE
            + m.group(2)
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

    print("\n=== Done ===")
    if not dry_run:
        print("Enable with: export VLLM_DSA_KV_OFFLOAD=1")
        print("Rollback:    python3 apply_patches.py", vllm_dir, "--rollback")


if __name__ == "__main__":
    main()
