import os

TIDALCACHE_ENABLED = os.environ.get("VLLM_DSA_KV_OFFLOAD", "0") == "1"
HUGEPAGE_PATH = os.environ.get("VLLM_DSA_OFFLOAD_HUGEPAGE_PATH", "/dev/hugepages")

_GLOBAL_MANAGER = None
