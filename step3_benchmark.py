#!/usr/bin/env python3
"""
step3_benchmark.py
==================
Experiment 3: End-to-End Latency & NVML Telemetry Suite

Runs the full benchmark pipeline:
  - Simulates RAG retrieval with parallel English / Amharic corpora
  - Measures fertility tax and KV-cache overhead per tokenizer
  - Records NVML telemetry snapshots (VRAM delta, temperature, power)
  - Simulates TTFT (Time-To-First-Token) and TPS (Tokens-Per-Second) via
    timing the tokenization + pruning stage as a proxy workload
  - Writes results to results/telemetry_logs.csv

Usage
-----
    python step3_benchmark.py

Dependencies
------------
    pip install transformers nvidia-ml-py tabulate tiktoken
"""

from __future__ import annotations

import sys
import io
# Force UTF-8 output so Ethiopic characters render correctly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import csv
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from tabulate import tabulate

# Optional GPU telemetry
try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

from step2_middleware import (
    estimate_fertility_ratio,
    get_gpu_telemetry,
    prune_chunks,
    select_quantization,
)

DATA_PATH = Path(__file__).parent / "data" / "parallel_corpus.json"
RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "telemetry_logs.csv"

TOKENIZER_IDS = [
    ("Qwen-2.5 (Multilingual BPE)", "Qwen/Qwen2.5-0.5B"),
    ("Phi-3.5-Mini (Byte-Fallback BPE)", "microsoft/Phi-3.5-mini-instruct"),
    ("SmolLM2 (Byte-Level BPE)", "HuggingFaceTB/SmolLM2-135M"),
]

NUM_RUNS = 3  # Repeat each experiment for stability


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    run: int
    tokenizer_label: str
    language: str
    num_chunks: int
    total_words: int
    fertility_ratio: float
    estimated_tokens: int
    pruned_chunks: int
    pruned_tokens: int
    quant_tier: str
    token_budget: int
    ttft_proxy_ms: float
    vram_free_mb: float
    vram_used_mb: float
    temperature_c: int
    power_draw_w: float
    throttle_reason: str


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def load_corpus() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {DATA_PATH}.\n"
            "Please ensure data/parallel_corpus.json is present."
        )
    with open(DATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def run_single_benchmark(
    run_id: int,
    tok_label: str,
    tokenizer,
    chunks: List[str],
    language: str,
) -> BenchmarkRow:
    """Run one benchmark pass: fertility + pruning + telemetry snapshot."""
    tel = get_gpu_telemetry()
    free_vram_mb = tel.free_vram_mb if tel else 8192.0
    quant_label, token_budget = select_quantization(free_vram_mb)

    # Fertility
    total_words = sum(len(c.split()) for c in chunks)
    fertility_ratios = [estimate_fertility_ratio(c, tokenizer) for c in chunks]
    avg_fertility = sum(fertility_ratios) / max(len(fertility_ratios), 1)
    estimated_tokens = int(math.ceil(total_words * avg_fertility))

    # Pruning with timing (proxy for TTFT preprocessing)
    t0 = time.perf_counter()
    pruned, pruned_tokens = prune_chunks(chunks, token_budget, tokenizer)
    t1 = time.perf_counter()
    ttft_proxy_ms = (t1 - t0) * 1000.0

    return BenchmarkRow(
        run=run_id,
        tokenizer_label=tok_label,
        language=language,
        num_chunks=len(chunks),
        total_words=total_words,
        fertility_ratio=round(avg_fertility, 3),
        estimated_tokens=estimated_tokens,
        pruned_chunks=len(pruned),
        pruned_tokens=pruned_tokens,
        quant_tier=quant_label,
        token_budget=token_budget,
        ttft_proxy_ms=round(ttft_proxy_ms, 2),
        vram_free_mb=round(tel.free_vram_mb, 1) if tel else -1.0,
        vram_used_mb=round(tel.used_vram_mb, 1) if tel else -1.0,
        temperature_c=tel.temperature_c if tel else -1,
        power_draw_w=round(tel.power_draw_w, 1) if tel else -1.0,
        throttle_reason=tel.throttle_reason if tel else "N/A",
    )


def main() -> None:
    print("=" * 70)
    print("  Ge'ez-EdgeRAG — Experiment 3: End-to-End Benchmark Suite")
    print("=" * 70)

    corpus = load_corpus()
    en_chunks = [e["english"] for e in corpus["entries"]]
    am_chunks = [e["amharic"] for e in corpus["entries"]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[BenchmarkRow] = []

    for tok_label, hf_id in TOKENIZER_IDS:
        print(f"\n  Loading tokenizer: {hf_id} ...", flush=True)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=True
            )
        except Exception as exc:
            print(f"  [WARN] Could not load {hf_id}: {exc} — skipping.")
            continue

        for run in range(1, NUM_RUNS + 1):
            print(f"    Run {run}/{NUM_RUNS}: English ...", end=" ", flush=True)
            en_row = run_single_benchmark(run, tok_label, tokenizer, en_chunks, "EN")
            rows.append(en_row)
            print(f"done ({en_row.ttft_proxy_ms:.1f} ms)")

            print(f"    Run {run}/{NUM_RUNS}: Amharic ...", end=" ", flush=True)
            am_row = run_single_benchmark(run, tok_label, tokenizer, am_chunks, "AM")
            rows.append(am_row)
            print(f"done ({am_row.ttft_proxy_ms:.1f} ms)")

    if not rows:
        print("\n[ERROR] No benchmark data collected. Check tokenizer availability.")
        return

    # Write CSV
    fieldnames = list(asdict(rows[0]).keys())
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"\n  Results written to: {LOG_PATH}")

    # Summary table
    print("\n" + "=" * 70)
    print("  Summary: Fertility Tax & Pruning Proxy-TTFT (mean across runs)")
    print("=" * 70)

    from collections import defaultdict
    summary: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        summary[row.tokenizer_label][row.language].append(row)

    table_rows = []
    for tok_label in summary:
        for lang in ("EN", "AM"):
            lang_rows = summary[tok_label][lang]
            if not lang_rows:
                continue
            avg_fertility = sum(r.fertility_ratio for r in lang_rows) / len(lang_rows)
            avg_tok = sum(r.estimated_tokens for r in lang_rows) / len(lang_rows)
            avg_ttft = sum(r.ttft_proxy_ms for r in lang_rows) / len(lang_rows)
            avg_pruned = sum(r.pruned_tokens for r in lang_rows) / len(lang_rows)
            table_rows.append([
                tok_label,
                lang,
                f"{avg_fertility:.2f}x",
                f"{avg_tok:.0f}",
                f"{avg_pruned:.0f}",
                f"{avg_ttft:.2f} ms",
            ])

    headers = [
        "Tokenizer", "Lang", "Avg Fertility",
        "Est. Tokens (raw)", "Pruned Tokens", "Proxy TTFT"
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print()


if __name__ == "__main__":
    main()
