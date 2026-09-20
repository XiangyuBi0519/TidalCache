import os
import logging

TIDALCACHE_ENABLED = os.environ.get("VLLM_DSA_KV_OFFLOAD", "0") == "1"
HUGEPAGE_PATH = os.environ.get("VLLM_DSA_OFFLOAD_HUGEPAGE_PATH", "/dev/hugepages")

_GLOBAL_MANAGER = None

logger = logging.getLogger("tidalcache")
logger.setLevel(logging.INFO)
logger.propagate = False
_log_path = os.environ.get("TIDALCACHE_LOG", "tidalcache.log")
_fh = logging.FileHandler(_log_path, mode="a")
_fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(process)d] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_fh)

# Path B: apply vllm kv_cache_config isolation patch (compress no longer shares
# raw_tensor with swa/state). Gated by TIDALCACHE_ISOLATE_COMPRESS=1.
try:
    from tidalcache.vllm_config_patch import apply_kv_config_isolation_patch as _apply_iso
    _apply_iso()
except Exception as _e:
    logger.warning("[init] failed to apply kv config isolation patch: %s", _e)
