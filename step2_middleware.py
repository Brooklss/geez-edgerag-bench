#!/usr/bin/env python3
"""
step2_middleware.py
===================
Experiment 2: VRAM-Aware Ge'ez Context Compression & Pruning Engine

This module implements the Ge'ez-EdgeRAG middleware that:

  1. Queries live NVML telemetry for free VRAM on the local GPU
  2. Estimates token fertility overhead for Ge'ez / Amharic chunks
  3. Scores and prunes low-density retrieval chunks (TF-IDF extractive)
  4. Routes to an appropriate quantization tier based on VRAM budget

Usage
-----
    python step2_middleware.py

Dependencies
------------
    pip install transformers nvidia-ml-py tabulate tiktoken
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from tabulate import tabulate

# Optional GPU telemetry — graceful CPU fallback
try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

DATA_PATH = Path(__file__).parent / "data" / "parallel_corpus.json"

# ---------------------------------------------------------------------------
# NVML Telemetry
# ---------------------------------------------------------------------------

@dataclass
class GPUTelemetry:
    device_name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float
    gpu_util_pct: int
    temperature_c: int
    power_draw_w: float
    throttle_reason: str


def get_gpu_telemetry(device_index: int = 0) -> Optional[GPUTelemetry]:
    """Query NVML for live GPU telemetry. Returns None on CPU-only machines."""
    if not _NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()

        # Throttle reason flags
        try:
            throttle_flags = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            if throttle_flags == 0:
                throttle_reason = "None"
            elif throttle_flags & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown:
                throttle_reason = "SW Thermal Slowdown"
            elif throttle_flags & pynvml.nvmlClocksThrottleReasonHwSlowdown:
                throttle_reason = "HW Slowdown"
            else:
                throttle_reason = f"Flags: 0x{throttle_flags:08X}"
        except Exception:
            throttle_reason = "N/A"

        return GPUTelemetry(
            device_name=name,
            total_vram_mb=mem.total / (1024 ** 2),
            used_vram_mb=mem.used / (1024 ** 2),
            free_vram_mb=mem.free / (1024 ** 2),
            gpu_util_pct=util.gpu,
            temperature_c=temp,
            power_draw_w=power,
            throttle_reason=throttle_reason,
        )
    except Exception as exc:
        warnings.warn(f"NVML telemetry error: {exc}")
        return None


def print_telemetry(tel: Optional[GPUTelemetry]) -> None:
    if tel is None:
        print("  [INFO] No NVIDIA GPU detected — running in CPU fallback mode.")
        return
    rows = [
        ["Device", tel.device_name],
        ["Total VRAM", f"{tel.total_vram_mb:.0f} MB"],
        ["Used VRAM", f"{tel.used_vram_mb:.0f} MB"],
        ["Free VRAM", f"{tel.free_vram_mb:.0f} MB"],
        ["GPU Utilisation", f"{tel.gpu_util_pct}%"],
        ["Temperature", f"{tel.temperature_c} °C"],
        ["Power Draw", f"{tel.power_draw_w:.1f} W"],
        ["Throttle Reason", tel.throttle_reason],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline"))


# ---------------------------------------------------------------------------
# Quantization router
# ---------------------------------------------------------------------------

QUANT_TIERS = [
    (6144, "Q8_0 (8-bit)",  4000),
    (4096, "Q4_K_M (4-bit)", 6000),
    (0,    "Q3_K (3-bit)",   8000),
]


def select_quantization(free_vram_mb: float) -> Tuple[str, int]:
    """Select quantization level and max Amharic token budget from free VRAM."""
    for threshold_mb, quant_label, token_budget in QUANT_TIERS:
        if free_vram_mb >= threshold_mb:
            return quant_label, token_budget
    return QUANT_TIERS[-1][1], QUANT_TIERS[-1][2]


# ---------------------------------------------------------------------------
# Fertility estimator
# ---------------------------------------------------------------------------

# Ethiopic Unicode block: U+1200–U+137F (and extended blocks)
_ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")


def estimate_fertility_ratio(text: str, tokenizer=None) -> float:
    """
    Estimate byte-fallback fertility for a chunk.

    If a tokenizer is provided, compute exact fertility ratio.
    Otherwise use the Ethiopic character density as a heuristic proxy:
        fertility_heuristic = 1.0 + 5.5 * ethiopic_density
    Rationale: pure Ethiopic text yields ~6.5x fertility on byte-fallback
    tokenizers; English (density ≈ 0) yields baseline ~1.0x.
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        return len(ids) / max(len(words), 1)

    chars = len(text)
    if chars == 0:
        return 1.0
    ethiopic_chars = len(_ETHIOPIC_RE.findall(text))
    density = ethiopic_chars / chars
    return 1.0 + 5.5 * density


# ---------------------------------------------------------------------------
# TF-IDF chunk pruner
# ---------------------------------------------------------------------------

def _tokenise_simple(text: str) -> List[str]:
    """Minimal whitespace + punctuation tokeniser for TF-IDF scoring."""
    return re.findall(r"\b\w+\b", text.lower())


def tfidf_scores(chunks: List[str]) -> List[float]:
    """Compute a TF-IDF relevance score for each chunk relative to the corpus."""
    # Build IDF
    N = len(chunks)
    df: Counter = Counter()
    chunk_tfs: List[Counter] = []
    for chunk in chunks:
        tokens = _tokenise_simple(chunk)
        tf = Counter(tokens)
        chunk_tfs.append(tf)
        df.update(set(tokens))

    idf = {term: math.log((N + 1) / (df[term] + 1)) + 1 for term in df}

    scores = []
    for tf in chunk_tfs:
        score = sum(tf[t] * idf.get(t, 0) for t in tf)
        scores.append(score)
    return scores


def prune_chunks(
    chunks: List[str],
    token_budget: int,
    tokenizer=None,
) -> Tuple[List[str], int]:
    """
    Prune low-TF-IDF chunks until the total estimated token count fits
    within `token_budget`.

    Returns (pruned_chunks, estimated_total_tokens).
    """
    # Score chunks
    scores = tfidf_scores(chunks)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    selected: List[str] = []
    total_tokens = 0

    for score, chunk in ranked:
        fertility = estimate_fertility_ratio(chunk, tokenizer)
        words = len(chunk.split())
        est_tokens = int(math.ceil(words * fertility))

        if total_tokens + est_tokens <= token_budget:
            selected.append(chunk)
            total_tokens += est_tokens
        else:
            # Try to fit a shortened version
            available = token_budget - total_tokens
            if available > 50:
                # Trim chunk to approximately `available` tokens
                trim_words = int(available / max(fertility, 1))
                trimmed = " ".join(chunk.split()[:trim_words])
                trim_est = int(math.ceil(len(trimmed.split()) * fertility))
                selected.append(trimmed)
                total_tokens += trim_est
            break

    return selected, total_tokens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_middleware(chunks: List[str]) -> None:
    print("=" * 70)
    print("  Ge'ez-EdgeRAG — Experiment 2: VRAM-Aware Context Pruner")
    print("=" * 70)

    # Step 1: GPU telemetry
    print("\n[1] Live NVML Telemetry\n")
    tel = get_gpu_telemetry()
    print_telemetry(tel)

    free_vram_mb = tel.free_vram_mb if tel else 8192.0  # assume 8 GB in fallback

    # Step 2: Quantization routing
    print("\n[2] Quantization Routing\n")
    quant_label, token_budget = select_quantization(free_vram_mb)
    print(f"  Free VRAM         : {free_vram_mb:.0f} MB")
    print(f"  Selected Tier     : {quant_label}")
    print(f"  Amharic Budget    : {token_budget:,} tokens")

    # Step 3: Fertility analysis per chunk
    print("\n[3] Per-Chunk Fertility Analysis\n")
    fertility_rows = []
    for i, chunk in enumerate(chunks, 1):
        fr = estimate_fertility_ratio(chunk)
        words = len(chunk.split())
        est_tok = int(math.ceil(words * fr))
        fertility_rows.append([
            f"Chunk {i}",
            words,
            f"{fr:.2f}x",
            est_tok,
            chunk[:60] + ("..." if len(chunk) > 60 else ""),
        ])
    print(tabulate(
        fertility_rows,
        headers=["Chunk", "Words", "Fertility", "Est. Tokens", "Preview"],
        tablefmt="github",
    ))

    # Step 4: Prune
    print("\n[4] TF-IDF Context Pruning\n")
    pruned, final_tokens = prune_chunks(chunks, token_budget)
    print(f"  Original chunks   : {len(chunks)}")
    print(f"  After pruning     : {len(pruned)}")
    print(f"  Total est. tokens : {final_tokens:,} / {token_budget:,} (budget)")

    print("\n[5] Pruned Context (ready for SLM inference)\n")
    for i, chunk in enumerate(pruned, 1):
        print(f"  --- Chunk {i} ---")
        print(f"  {chunk[:200]}")
        print()

    print("=" * 70)
    print(f"  Middleware complete. Routing to {quant_label} inference.")
    print("=" * 70)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {DATA_PATH}.\n"
            "Please ensure data/parallel_corpus.json is present."
        )

    with open(DATA_PATH, encoding="utf-8") as fh:
        corpus_data = json.load(fh)

    # Use all Amharic chunks from corpus as simulated retrieved documents
    chunks = [entry["amharic"] for entry in corpus_data["entries"]]
    run_middleware(chunks)


if __name__ == "__main__":
    main()
