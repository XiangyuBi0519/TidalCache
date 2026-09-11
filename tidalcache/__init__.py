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
