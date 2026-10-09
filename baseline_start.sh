#!/bin/bash
# ============================================================================
# BASELINE launch — TidalCache OFF, graph mode ON.
#
# Purpose: apples-to-apples comparison against the TidalCache run. Identical
# vllm flags to the TidalCache start.sh EXCEPT the TidalCache env switches are
# cleared, so kv_offload_enabled=False and every gather/scatter branch is
# skipped (behaves as stock vllm-ascend). CP_PATCH0 (graph-params shim) stays
# active in the patched source — it is required to run graph mode at all and is
# a no-op, so it does not affect the baseline.
#
# IMPORTANT: keep EVERY vllm flag below byte-identical to your TidalCache
# start.sh (same model, TP/DP, max-model-len, max-num-seqs, block-size,
# speculative-config, additional-config, etc.). The ONLY difference between the
# two runs must be the TidalCache env block. Otherwise the comparison is noise.
# ============================================================================

# ── TidalCache OFF ──────────────────────────────────────────────────────────
unset VLLM_DSA_KV_OFFLOAD          # master switch (TIDALCACHE_ENABLED reads this)
unset TIDALCACHE_HOST_COMPRESS
unset TIDALCACHE_ATTN_ON_SEL
unset TIDALCACHE_CSA_ONLY
unset TIDALCACHE_PREFILL_MODE
unset TIDALCACHE_POISON
# ─────────────────────────────────────────────────────────────────────────────

nohup python -m vllm.entrypoints.openai.api_server \
        --model /mnt/paas/kubernetes/kubelet/DeepSeek-V4-Flash-w8a8-mtp \
        --max-model-len 133120 \
        --max-num-batched-tokens 4096 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 16 \
        --data-parallel-size 2 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --no-enable-prefix-caching \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
        --quantization ascend \
        --port 8900 \
        --block-size 128 \
        --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": false}' \
        --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true, "multistream_overlap_shared_expert": true, "enable_npugraph_ex": true}' \
        > dsv4_service_baseline.log 2>&1 &

echo "baseline (TidalCache OFF, graph ON) launching → dsv4_service_baseline.log"
echo "wait for 'Application startup complete', then verify HBM is HIGHER than TidalCache run (compress KV stays on HBM)."
