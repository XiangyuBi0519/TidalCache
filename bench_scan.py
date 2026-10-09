#!/usr/bin/env python3
"""
TidalCache scan benchmark — sweeps context length and concurrency to quantify
the net trade-off: capacity gained (HBM freed -> more concurrency / longer ctx)
vs latency cost (per-step gather adds a Host->Device DMA hop).

Run the SAME command once per config (restart the server in between):
    # TidalCache ON  (graph mode, VLLM_DSA_KV_OFFLOAD=1 + TIDALCACHE_* set)
    python bench_scan.py --mode latency    --label tidalcache --csv scan.csv
    python bench_scan.py --mode throughput --label tidalcache --csv scan.csv
    # restart with baseline_start.sh (TidalCache OFF), then:
    python bench_scan.py --mode latency    --label baseline   --csv scan.csv
    python bench_scan.py --mode throughput --label baseline   --csv scan.csv

Then diff the two labels in scan.csv.

Modes:
  latency    : batch=1, sweep --ctx-list, fixed --out. Isolates per-request
               TTFT (prefill) and TPOT (decode, where gather cost shows up).
  throughput : fixed --ctx + --out, sweep --conc-list. Launches N concurrent
               requests, reports aggregate tok/s + per-request TTFT/TPOT
               p50/p99 + ok/fail. Ramp until failures = capacity ceiling.

Every config captures a short output tail for an eyeball correctness check and
flags empty/garbled/error responses.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests


# ── Controlled-length prompt builder ────────────────────────────────────────
# A coherent filler paragraph so long context carries real attention signal
# (not just padding). We repeat it to reach the target char count, then wrap
# with a summary instruction so the model must attend across the whole span.
_FILLER = (
    "人工智能的发展经历了多个阶段，从早期的符号推理到如今的大规模神经网络。"
    "深度学习的兴起让模型能够从海量数据中自动学习特征，推动了自然语言处理、"
    "计算机视觉和语音识别等领域的突破。随着模型规模的增长，算力和显存成为关键瓶颈。"
)


def build_prompt(target_chars: int) -> str:
    if target_chars <= 64:
        return "请用一句话解释什么是人工智能。"
    reps = max(1, target_chars // len(_FILLER))
    body = _FILLER * reps
    return f"请阅读下面这段文字，然后用三句话总结它的主要观点。\n\n{body}\n\n请给出你的三句话总结："


# ── Streaming request with per-chunk timestamps ─────────────────────────────
def stream_request(url: str, model: str, prompt: str, max_tokens: int,
                   timeout: int = 600) -> dict[str, Any]:
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
    token_times: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    text_tail = ""
    err = None
    try:
        with requests.post(url, json=payload,
                           headers={"Content-Type": "application/json"},
                           stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
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
                        text_tail = (text_tail + content)[-200:]
                usage = obj.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as e:  # noqa: BLE001 — record, don't crash the sweep
        err = f"{type(e).__name__}: {e}"

    total = time.time() - start
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
        "chunks": len(token_times),
        "tail": text_tail,
        "err": err,
    }


def pct(vals: list[float], p: float) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def fmt(v, unit="", nd=1):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


CSV_FIELDS = [
    "label", "mode", "ctx_chars", "concurrency", "prompt_tok", "out_tok",
    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
    "agg_tok_s", "ok", "fail", "sample_tail",
]


def write_csv_row(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_concurrent(url, model, prompt, out_tokens, concurrency):
    """Launch `concurrency` identical requests in parallel; return aggregate."""
    results = []
    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(stream_request, url, model, prompt, out_tokens)
                for _ in range(concurrency)]
        for fu in as_completed(futs):
            results.append(fu.result())
    wall = time.time() - wall_start
    ok = [r for r in results if r["err"] is None and r["completion_tokens"] > 0]
    fail = [r for r in results if r not in ok]
    agg_tok = sum(r["completion_tokens"] for r in ok)
    return {
        "wall_s": wall,
        "agg_tok_s": agg_tok / wall if wall else None,
        "ttft_p50": pct([r["ttft_ms"] for r in ok], 0.50),
        "ttft_p99": pct([r["ttft_ms"] for r in ok], 0.99),
        "tpot_p50": pct([r["tpot_ms"] for r in ok], 0.50),
        "tpot_p99": pct([r["tpot_ms"] for r in ok], 0.99),
        "ok": len(ok),
        "fail": len(fail),
        "prompt_tok": ok[0]["prompt_tokens"] if ok else 0,
        "tail": ok[0]["tail"] if ok else (fail[0]["err"] if fail else ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8900/v1/chat/completions")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--label", required=True, help="config label: baseline | tidalcache")
    ap.add_argument("--mode", choices=["latency", "throughput"], required=True)
    ap.add_argument("--csv", default="scan.csv", help="append results here")
    ap.add_argument("--warmup", type=int, default=1)
    # latency mode
    ap.add_argument("--ctx-list", default="256,1024,4096,8192,16384",
                    help="latency: context char targets to sweep")
    ap.add_argument("--lat-out", type=int, default=256, help="latency: output tokens")
    ap.add_argument("--lat-runs", type=int, default=3, help="latency: timed runs per ctx")
    # throughput mode
    ap.add_argument("--conc-list", default="1,2,4,8,16,32",
                    help="throughput: concurrency levels to sweep")
    ap.add_argument("--thr-ctx", type=int, default=2048, help="throughput: context chars")
    ap.add_argument("--thr-out", type=int, default=512, help="throughput: output tokens")
    ap.add_argument("--thr-repeats", type=int, default=1,
                    help="throughput: repeat each concurrency level N times, report median (de-noise)")
    args = ap.parse_args()

    print("=" * 108)
    print(f"scan benchmark — label={args.label}  mode={args.mode}  url={args.url}")
    print(f"csv -> {args.csv}")
    print("=" * 108)

    if args.mode == "latency":
        ctx_list = [int(x) for x in args.ctx_list.split(",") if x.strip()]
        print(f"\n{'ctx_chars':>10s} {'prompt_tok':>10s} {'out_tok':>8s} "
              f"{'TTFT_ms':>10s} {'TPOT_ms':>10s} {'tok/s':>8s}  sample_tail")
        print("-" * 108)
        for ctx in ctx_list:
            prompt = build_prompt(ctx)
            for _ in range(args.warmup):
                stream_request(args.url, args.model, prompt, args.lat_out)
            runs = [stream_request(args.url, args.model, prompt, args.lat_out)
                    for _ in range(args.lat_runs)]
            ok = [r for r in runs if r["err"] is None and r["completion_tokens"] > 0]
            if not ok:
                print(f"{ctx:>10d}  ALL FAILED: {runs[0]['err']}")
                write_csv_row(args.csv, {"label": args.label, "mode": "latency",
                                         "ctx_chars": ctx, "concurrency": 1,
                                         "fail": len(runs), "ok": 0,
                                         "sample_tail": runs[0]["err"]})
                continue
            ttft = statistics.median([r["ttft_ms"] for r in ok])
            tpot = statistics.median([r["tpot_ms"] for r in ok if r["tpot_ms"]])
            total = statistics.median([r["total_s"] for r in ok])
            ct = ok[0]["completion_tokens"]
            pt = ok[0]["prompt_tokens"]
            tps = ct / total if total else None
            tail = ok[0]["tail"].replace("\n", " ")[-40:]
            print(f"{ctx:>10d} {pt:>10d} {ct:>8d} {fmt(ttft):>10s} "
                  f"{fmt(tpot):>10s} {fmt(tps):>8s}  {tail}")
            write_csv_row(args.csv, {
                "label": args.label, "mode": "latency", "ctx_chars": ctx,
                "concurrency": 1, "prompt_tok": pt, "out_tok": ct,
                "ttft_p50_ms": round(ttft, 1), "tpot_p50_ms": round(tpot, 1) if tpot else "",
                "agg_tok_s": round(tps, 1) if tps else "", "ok": len(ok), "fail": len(runs) - len(ok),
                "sample_tail": ok[0]["tail"].replace("\n", " ")[-80:],
            })

    else:  # throughput
        conc_list = [int(x) for x in args.conc_list.split(",") if x.strip()]
        prompt = build_prompt(args.thr_ctx)
        print(f"\nctx={args.thr_ctx} chars, out={args.thr_out} tok")
        print(f"\n{'conc':>5s} {'ok/fail':>8s} {'agg_tok/s':>10s} "
              f"{'TTFT_p50':>9s} {'TTFT_p99':>9s} {'TPOT_p50':>9s} {'TPOT_p99':>9s}  sample_tail")
        print("-" * 108)
        for _ in range(args.warmup):
            stream_request(args.url, args.model, prompt, args.thr_out)
        for conc in conc_list:
            reps = [run_concurrent(args.url, args.model, prompt, args.thr_out, conc)
                    for _ in range(max(1, args.thr_repeats))]
            # median across repeats for the rate/latency fields; sum ok/fail
            def _med(key):
                vals = [x[key] for x in reps if x[key] is not None]
                return statistics.median(vals) if vals else None
            r = {
                "agg_tok_s": _med("agg_tok_s"),
                "ttft_p50": _med("ttft_p50"), "ttft_p99": _med("ttft_p99"),
                "tpot_p50": _med("tpot_p50"), "tpot_p99": _med("tpot_p99"),
                "ok": sum(x["ok"] for x in reps), "fail": sum(x["fail"] for x in reps),
                "prompt_tok": next((x["prompt_tok"] for x in reps if x["prompt_tok"]), 0),
                "tail": next((x["tail"] for x in reps if x["tail"]), ""),
            }
            tail = (r["tail"] or "").replace("\n", " ")[-40:]
            print(f"{conc:>5d} {str(r['ok'])+'/'+str(r['fail']):>8s} "
                  f"{fmt(r['agg_tok_s']):>10s} {fmt(r['ttft_p50']):>9s} "
                  f"{fmt(r['ttft_p99']):>9s} {fmt(r['tpot_p50']):>9s} "
                  f"{fmt(r['tpot_p99']):>9s}  {tail}")
            write_csv_row(args.csv, {
                "label": args.label, "mode": "throughput", "ctx_chars": args.thr_ctx,
                "concurrency": conc, "prompt_tok": r["prompt_tok"], "out_tok": args.thr_out,
                "ttft_p50_ms": round(r["ttft_p50"], 1) if r["ttft_p50"] else "",
                "ttft_p99_ms": round(r["ttft_p99"], 1) if r["ttft_p99"] else "",
                "tpot_p50_ms": round(r["tpot_p50"], 1) if r["tpot_p50"] else "",
                "tpot_p99_ms": round(r["tpot_p99"], 1) if r["tpot_p99"] else "",
                "agg_tok_s": round(r["agg_tok_s"], 1) if r["agg_tok_s"] else "",
                "ok": r["ok"], "fail": r["fail"],
                "sample_tail": (r["tail"] or "").replace("\n", " ")[-80:],
            })
            if r["fail"] > 0:
                print(f"       ^ {r['fail']} failures at conc={conc} — likely capacity ceiling")

    print("\ndone. compare labels with:  python bench_scan.py --help  (or inspect", args.csv, ")")


if __name__ == "__main__":
    main()
