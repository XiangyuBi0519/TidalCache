#!/usr/bin/env python3
"""
TidalCache performance benchmark: TTFT + TPOT across various request patterns.

Usage:
    python bench_tidalcache.py [--url URL] [--model MODEL] [--runs N] [--label LABEL]

Compare configs by running twice with different labels & piping output:
    python bench_tidalcache.py --label baseline > bench_baseline.txt
    # switch env vars, restart service
    python bench_tidalcache.py --label path_b > bench_path_b.txt

Metrics:
  - TTFT: time from send to first token (ms)
  - TPOT: mean time between consecutive tokens AFTER the first (ms)
  - Total: end-to-end wall time (s)
  - tok/s: completion_tokens / total_time
  - Runs the case `--runs` times (default 3) after `--warmup` unrecorded warmup
    runs; reports the median.
"""

import argparse
import json
import statistics
import sys
import time
from typing import Any, Optional

import requests


# ── Test cases: (name, prompt_builder, max_tokens) ───────────────────────
def _repeat(prompt: str, n: int) -> str:
    return prompt * n


TEST_CASES = [
    # name, prompt, max_tokens — cover short/medium/long prompt & output combos
    ("short_prompt_short_out",   "你好，请用一句话介绍自己。",                              32),
    ("short_prompt_medium_out",  "解释一下什么是牛顿第一定律。",                            256),
    ("short_prompt_long_out",    "你好，能写一篇1000字的科幻小说么？",                      1024),
    ("code_gen_medium_out",      "写一个Python函数实现快速排序，包含详细注释和多种测试用例。", 512),
    ("math_reasoning_medium",    "如果一个三角形三条边分别是3、4、5，请判断它是否是直角三角形并详细说明理由。", 256),
    ("long_prompt_short_out",    _repeat("请解释一下量子计算的基本原理。", 100),             128),
    ("long_prompt_medium_out",   _repeat("请解释一下量子计算的基本原理。", 100),             512),
]


# ── Streaming request with per-chunk timestamps ─────────────────────────
def stream_request(
    url: str, model: str, prompt: str, max_tokens: int, timeout: int = 300,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.time()
    ttft: Optional[float] = None
    token_times: list[float] = []  # timestamps of received content chunks
    completion_tokens = 0
    prompt_tokens = 0

    with requests.post(
        url, json=payload,
        headers={"Content-Type": "application/json"},
        stream=True, timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):]
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    now = time.time()
                    if ttft is None:
                        ttft = now - start
                    token_times.append(now)

            # Final chunk may carry usage stats
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

    total = time.time() - start

    # TPOT: mean gap between consecutive content chunks, skipping the first
    # (which is captured by TTFT).
    if len(token_times) >= 2:
        gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        tpot = sum(gaps) / len(gaps)
    else:
        tpot = None

    return {
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "tpot_ms": tpot * 1000 if tpot is not None else None,
        "total_s": total,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "chunks_seen": len(token_times),
    }


def fmt(v, unit: str = "") -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.1f}{unit}"
    return f"{v}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8900/v1/chat/completions")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per case")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warmup runs")
    ap.add_argument("--label", default="", help="config label for header/output")
    ap.add_argument("--sleep-between", type=float, default=0.5, help="sleep between runs")
    ap.add_argument("--filter", default="", help="only run cases whose name contains this")
    args = ap.parse_args()

    print("=" * 100)
    print(f"TidalCache benchmark — label={args.label or '(unset)'}, url={args.url}, model={args.model}")
    print(f"runs per test: {args.runs} timed + {args.warmup} warmup, sleep between: {args.sleep_between}s")
    print("=" * 100)

    header = (f"\n{'test':30s} {'prompt_tok':>10s} {'out_tok':>8s} "
              f"{'TTFT_ms':>10s} {'TPOT_ms':>10s} {'total_s':>10s} {'tok/s':>8s}")
    print(header)
    print("-" * 100)

    all_results = {}
    cases = [c for c in TEST_CASES if args.filter in c[0]] if args.filter else TEST_CASES

    for name, prompt, max_tokens in cases:
        for _ in range(args.warmup):
            try:
                stream_request(args.url, args.model, prompt, max_tokens)
            except Exception as e:
                print(f"[warmup] {name} failed: {e}", file=sys.stderr)
            time.sleep(args.sleep_between)

        results = []
        for run_idx in range(args.runs):
            try:
                r = stream_request(args.url, args.model, prompt, max_tokens)
                results.append(r)
            except Exception as e:
                print(f"[run {run_idx}] {name} failed: {e}", file=sys.stderr)
            time.sleep(args.sleep_between)

        if not results:
            print(f"{name:30s} {'N/A':>10s} {'N/A':>8s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>8s}")
            continue

        def med(key):
            vals = [r[key] for r in results if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        ttft = med("ttft_ms")
        tpot = med("tpot_ms")
        total = med("total_s")
        pt = results[0]["prompt_tokens"]
        ct = results[0]["completion_tokens"]
        toks_per_s = ct / total if total else None

        all_results[name] = {
            "ttft_ms": ttft, "tpot_ms": tpot, "total_s": total,
            "prompt_tokens": pt, "completion_tokens": ct,
            "toks_per_s": toks_per_s, "runs": len(results),
        }

        print(f"{name:30s} {pt:>10d} {ct:>8d} "
              f"{fmt(ttft):>10s} {fmt(tpot):>10s} {fmt(total):>10s} {fmt(toks_per_s):>8s}")

    print("-" * 100)
    print("\n=== Summary ===")
    if all_results:
        avg_ttft = statistics.mean([v["ttft_ms"] for v in all_results.values() if v["ttft_ms"]])
        avg_tpot = statistics.mean([v["tpot_ms"] for v in all_results.values() if v["tpot_ms"]])
        avg_tps  = statistics.mean([v["toks_per_s"] for v in all_results.values() if v["toks_per_s"]])
        print(f"avg TTFT: {avg_ttft:.1f} ms")
        print(f"avg TPOT: {avg_tpot:.1f} ms  ({1000/avg_tpot:.1f} tok/s decode)")
        print(f"avg throughput: {avg_tps:.1f} tok/s (end-to-end)")

    print("\n=== JSON (for diff) ===")
    print(json.dumps({
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "results": all_results,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
