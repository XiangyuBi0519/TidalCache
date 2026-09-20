"""
Path B: isolate compress from swa/state at vllm's KV cache config generation.

vllm's `_get_kv_cache_config_deepseek_v4` packs multiple KV cache groups into
shared raw_tensors — layers from compress + swa + state at the same slot_idx
end up backing onto ONE `KVCacheTensor.shared_by` list. This makes B3 v2
compress replacement corrupt state/that require Device HBM semantics.

Path B: monkey-patch the two involved functions so that compress groups (whose
layer names end in `.self_attn.attn`) get their own bucketing pass, guaranteeing
each compress `KVCacheTensor.shared_by` contains ONLY compress layers.

Activation: env `TIDALCACHE_ISOLATE_COMPRESS=1`.
"""

import logging
import os

logger = logging.getLogger("tidalcache")

_COMPRESS_SUFFIX = ".self_attn.attn"


def _group_is_compress(group) -> bool:
    return any(ln.endswith(_COMPRESS_SUFFIX) for ln in group.layer_names)


def apply_kv_config_isolation_patch() -> bool:
    """Monkey-patch vllm.v1.core.kv_cache_utils for DSV4 compress isolation.

    Returns True if patched, False if skipped (disabled / already patched /
    vllm not importable / target functions not present).
    """
    if os.environ.get("TIDALCACHE_ISOLATE_COMPRESS", "0") != "1":
        return False

    try:
        from vllm.v1.core import kv_cache_utils
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    except ImportError as e:
        logger.warning("[isolate-compress] vllm import failed: %s", e)
        return False

    if getattr(kv_cache_utils, "_tidalcache_isolate_patched", False):
        return False

    # Verify target functions are present. Fail-open if vllm's API changed.
    for name in (
        "_get_kv_cache_config_deepseek_v4",
        "_pool_bytes_per_block",
        "_bucket_layers_by_page_size",
        "KVCacheTensor",
        "may_override_num_blocks",
    ):
        if not hasattr(kv_cache_utils, name):
            logger.warning(
                "[isolate-compress] vllm.kv_cache_utils missing %s — vllm API changed?",
                name,
            )
            return False

    _original_deepseek_v4 = kv_cache_utils._get_kv_cache_config_deepseek_v4
    _original_pool_bytes = kv_cache_utils._pool_bytes_per_block
    _bucket = kv_cache_utils._bucket_layers_by_page_size
    _KVCacheTensor = kv_cache_utils.KVCacheTensor
    _may_override = kv_cache_utils.may_override_num_blocks

    def _patched_deepseek_v4(vllm_config, kv_cache_groups, available_memory):
        compress_groups = [g for g in kv_cache_groups if _group_is_compress(g)]
        other_groups = [g for g in kv_cache_groups if not _group_is_compress(g)]

        # No compress layers → passthrough to original (nothing to isolate).
        if not compress_groups:
            return _original_deepseek_v4(vllm_config, kv_cache_groups, available_memory)

        compress_buckets = _bucket(compress_groups)
        other_buckets = _bucket(other_groups) if other_groups else {}

        # Compute total bytes per block across BOTH bucket sets. Each raw_tensor
        # gets `ps * num_blocks` bytes; sum contributions from both compress
        # and other buckets to derive num_blocks.
        total_bytes_per_block = (
            sum(ps * len(slots) for ps, slots in compress_buckets.items())
            + sum(ps * len(slots) for ps, slots in other_buckets.items())
        )
        num_blocks = available_memory // total_bytes_per_block
        num_blocks = _may_override(vllm_config, num_blocks)

        kv_cache_tensors = []
        n_compress_tensors = 0
        n_other_tensors = 0
        for ps, slots in compress_buckets.items():
            for slot in slots:
                kv_cache_tensors.append(_KVCacheTensor(size=ps * num_blocks, shared_by=slot))
                n_compress_tensors += 1
        for ps, slots in other_buckets.items():
            for slot in slots:
                kv_cache_tensors.append(_KVCacheTensor(size=ps * num_blocks, shared_by=slot))
                n_other_tensors += 1

        logger.info(
            "[isolate-compress] deepseek_v4 config: compress-only raw_tensors=%d, "
            "other raw_tensors=%d, num_blocks=%d, total_bytes_per_block=%d",
            n_compress_tensors, n_other_tensors, num_blocks, total_bytes_per_block,
        )
        return num_blocks, kv_cache_tensors

    def _patched_pool_bytes(kv_cache_groups):
        # Match original's uniform-spec early return.
        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            return kv_cache_groups[0].kv_cache_spec.page_size_bytes
        if all(
            isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) for g in kv_cache_groups
        ):
            compress_groups = [g for g in kv_cache_groups if _group_is_compress(g)]
            other_groups = [g for g in kv_cache_groups if not _group_is_compress(g)]
            if compress_groups:
                compress_buckets = _bucket(compress_groups)
                other_buckets = _bucket(other_groups) if other_groups else {}
                return (
                    sum(ps * len(slots) for ps, slots in compress_buckets.items())
                    + sum(ps * len(slots) for ps, slots in other_buckets.items())
                )
        # No compress, or non-uniform specs — fall through to original behavior.
        return _original_pool_bytes(kv_cache_groups)

    kv_cache_utils._get_kv_cache_config_deepseek_v4 = _patched_deepseek_v4
    kv_cache_utils._pool_bytes_per_block = _patched_pool_bytes
    kv_cache_utils._tidalcache_isolate_patched = True

    logger.info(
        "[isolate-compress] Path B active — patched "
        "_get_kv_cache_config_deepseek_v4 and _pool_bytes_per_block"
    )
    return True
