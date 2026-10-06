#!/usr/bin/env python3
"""
step1_token_tax.py
==================
Experiment 1: Intrinsic Tokenizer Fertility & KV-Cache Memory Profiler

Measures subword fertility ratios and projected KV-cache memory consumption
across three SLM tokenizer architectures on parallel English-Amharic text:

  - Qwen-2.5-0.5B  (multilingual BPE, large Ethiopic vocabulary coverage)
  - Phi-3.5-Mini   (byte-fallback BPE, limited Ethiopic coverage)
  - SmolLM2-135M   (byte-level BPE, full UTF-8 fallback)

Usage
-----
    python step1_token_tax.py

Dependencies
------------
    pip install transformers tiktoken tabulate
"""

from __future__ import annotations

import sys
import io
# Force UTF-8 output so Ethiopic characters render correctly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from tabulate import tabulate
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "parallel_corpus.json"

# Hugging Face model IDs to benchmark
TOKENIZER_CONFIGS = [
    {
        "label": "Qwen-2.5 (Multilingual BPE)",
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "num_layers": 24,
        "num_kv_heads": 8,
        "head_dim": 64,
        "dtype_bytes": 2,  # float16
    },
    {
        "label": "Phi-3.5-Mini (Byte-Fallback BPE)",
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 96,
        "dtype_bytes": 2,
    },
    {
        "label": "SmolLM2 (Byte-Level BPE)",
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "num_layers": 30,
        "num_kv_heads": 4,
        "head_dim": 64,
        "dtype_bytes": 2,
    },
]

NUM_RAG_CHUNKS = 5          # Standard RAG top-K retrieval count
TOKENS_PER_CHUNK_EN = 320   # Approximate tokens per English chunk


# ---------------------------------------------------------------------------
# KV-Cache memory estimation
# ---------------------------------------------------------------------------

def kv_cache_bytes(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """
    Compute KV-cache VRAM (bytes) for a given token count.

    Formula:
        KV_mem = 2 × L × H_kv × d × t × dtype_bytes
    where:
        L          = number of transformer layers
        H_kv       = number of key/value attention heads
        d          = head dimension
        t          = sequence length in tokens
        dtype_bytes= bytes per element (2 for FP16, 1 for INT8)
    """
    return 2 * num_layers * num_kv_heads * head_dim * num_tokens * dtype_bytes


def bytes_to_mb(n: int) -> float:
    return n / (1024 ** 2)


# ---------------------------------------------------------------------------
# Fertility analysis
# ---------------------------------------------------------------------------

@dataclass
class FertilityResult:
    label: str
    en_tokens: int
    am_tokens: int
    fertility_ratio: float
    en_context_tokens: int
    am_context_tokens: int
    en_context_mb: float
    am_context_mb: float


def analyse_tokenizer(config: dict, corpus: dict) -> FertilityResult:
    """Load tokenizer, tokenize parallel sentences, compute fertility metrics."""
    print(f"  Loading tokenizer: {config['hf_id']} ...", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["hf_id"],
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer '{config['hf_id']}': {exc}\n"
            "Ensure you have internet access and `transformers` installed."
        ) from exc

    en_text = corpus["english"]
    am_text = corpus["amharic"]

    en_ids = tokenizer.encode(en_text, add_special_tokens=False)
    am_ids = tokenizer.encode(am_text, add_special_tokens=False)

    en_tok = len(en_ids)
    am_tok = len(am_ids)
    ratio = am_tok / max(en_tok, 1)

    # Project to NUM_RAG_CHUNKS context window
    en_ctx = TOKENS_PER_CHUNK_EN * NUM_RAG_CHUNKS
    am_ctx = int(math.ceil(en_ctx * ratio))

    en_mb = bytes_to_mb(
        kv_cache_bytes(
            en_ctx,
            config["num_layers"],
            config["num_kv_heads"],
            config["head_dim"],
            config["dtype_bytes"],
        )
    )
    am_mb = bytes_to_mb(
        kv_cache_bytes(
            am_ctx,
            config["num_layers"],
            config["num_kv_heads"],
            config["head_dim"],
            config["dtype_bytes"],
        )
    )

    return FertilityResult(
        label=config["label"],
        en_tokens=en_tok,
        am_tokens=am_tok,
        fertility_ratio=ratio,
        en_context_tokens=en_ctx,
        am_context_tokens=am_ctx,
        en_context_mb=en_mb,
        am_context_mb=am_mb,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Ge'ez-EdgeRAG — Experiment 1: Tokenizer Fertility & KV-Cache Tax")
    print("=" * 70)

    # Load corpus
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {DATA_PATH}.\n"
            "Please ensure data/parallel_corpus.json is present."
        )

    with open(DATA_PATH, encoding="utf-8") as fh:
        corpus_data = json.load(fh)

    # Use first entry as the reference sentence pair
    entry = corpus_data["entries"][0]

    print(f"\nReference English  : {entry['english'][:80]}...")
    print(f"Reference Amharic  : {entry['amharic'][:80]}...")
    print()

    results: List[FertilityResult] = []
    for cfg in TOKENIZER_CONFIGS:
        try:
            result = analyse_tokenizer(cfg, entry)
            results.append(result)
        except Exception as exc:
            print(f"  [WARN] Skipped {cfg['label']}: {exc}")

    if not results:
        print("\n[ERROR] No tokenizers could be evaluated. Check your environment.")
        return

    # Build table
    headers = [
        "Tokenizer Architecture",
        "EN Tokens",
        "AM Tokens",
        "Fertility Tax",
        "5-Chunk Context (EN)",
        "5-Chunk Context (AM)",
    ]
    rows = [
        [
            r.label,
            r.en_tokens,
            r.am_tokens,
            f"{r.fertility_ratio:.2f}x",
            f"{r.en_context_tokens} tok ({r.en_context_mb:.1f} MB)",
            f"{r.am_context_tokens} tok ({r.am_context_mb:.1f} MB)",
        ]
        for r in results
    ]

    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    print("\nNote: KV-cache memory estimated at float16 precision.")
    print("      VRAM ceiling: 8 GB (NVIDIA RTX 5060 Laptop GPU)")


if __name__ == "__main__":
    main()
