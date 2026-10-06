#!/usr/bin/env python3
"""
tests/test_middleware.py
========================
Unit tests for the Ge'ez-EdgeRAG middleware components.

Designed for CPU-only execution (no GPU required).
Run with:  pytest tests/ -v

Tests cover:
  - Ethiopic fertility heuristic
  - TF-IDF chunk scoring
  - Chunk pruner (fits budget / trims correctly)
  - KV-cache memory estimation
  - Quantization tier routing
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step2_middleware import (
    estimate_fertility_ratio,
    kv_cache_bytes,
    prune_chunks,
    select_quantization,
    tfidf_scores,
)


# ---------------------------------------------------------------------------
# Fertility estimator tests
# ---------------------------------------------------------------------------

class TestFertilityEstimator:
    def test_pure_english_near_one(self):
        """English text should have fertility close to 1.0x (heuristic)."""
        text = "The quick brown fox jumps over the lazy dog"
        ratio = estimate_fertility_ratio(text)
        assert 0.9 <= ratio <= 1.5, f"Expected ~1.0 for English, got {ratio}"

    def test_pure_amharic_elevated(self):
        """Pure Amharic (Ethiopic) text should yield fertility >> 1.0."""
        text = "ሪትሪቫል-ኦጎሜንትድ ጀነሬሽን ዘዴ ነው ፈላጊው ዳታቤዝ ያሰባስባል"
        ratio = estimate_fertility_ratio(text)
        assert ratio > 2.0, f"Expected >2.0 for Amharic, got {ratio}"

    def test_empty_string(self):
        """Empty string should return 1.0 (no inflation)."""
        assert estimate_fertility_ratio("") == 1.0

    def test_mixed_script_intermediate(self):
        """Mixed English-Amharic text should have intermediate fertility."""
        text = "RAG pipeline ሪትሪቫል ዘዴ vector database"
        ratio = estimate_fertility_ratio(text)
        en_ratio = estimate_fertility_ratio("RAG pipeline vector database")
        am_ratio = estimate_fertility_ratio("ሪትሪቫል ዘዴ ሲሆን ቀዳሚ ምእራፍ ስለ ሁኔታዎች")
        assert en_ratio < ratio < am_ratio, (
            f"Mixed ({ratio}) should be between EN ({en_ratio}) and AM ({am_ratio})"
        )

    def test_fertility_monotonic_with_ethiopic_density(self):
        """Fertility should increase as proportion of Ethiopic chars increases."""
        texts = [
            "Completely English text with no Ethiopic characters at all",
            "Some Ethiopic ሪትሪቫል mixed with English content here",
            "ሪትሪቫል ዳታቤዝ ሞዴሉ ቋንቋ ሁኔታዎች ምእራፍ ተዛምዷ ሰነድ",
        ]
        ratios = [estimate_fertility_ratio(t) for t in texts]
        assert ratios[0] < ratios[1] < ratios[2], (
            f"Fertility should increase with Ethiopic density: {ratios}"
        )


# ---------------------------------------------------------------------------
# TF-IDF scorer tests
# ---------------------------------------------------------------------------

class TestTfidfScores:
    def test_returns_correct_length(self):
        chunks = ["Hello world", "Foo bar baz", "Unique distinct content"]
        scores = tfidf_scores(chunks)
        assert len(scores) == len(chunks)

    def test_all_scores_non_negative(self):
        chunks = ["a b c", "d e f", "g h i"]
        scores = tfidf_scores(chunks)
        assert all(s >= 0 for s in scores), f"Got negative scores: {scores}"

    def test_unique_chunk_scores_higher(self):
        """A chunk with unique vocabulary should score higher than a duplicate."""
        chunks = [
            "common word repeated common word repeated",
            "common word repeated common word repeated",
            "unique rare vocabulary distinctive content never seen",
        ]
        scores = tfidf_scores(chunks)
        # Third chunk has unique terms, should have highest TF-IDF
        assert scores[2] > scores[0], (
            f"Unique chunk ({scores[2]}) should outscore duplicate ({scores[0]})"
        )

    def test_single_chunk(self):
        chunks = ["only one chunk"]
        scores = tfidf_scores(chunks)
        assert len(scores) == 1
        assert scores[0] >= 0

    def test_empty_chunks_list(self):
        assert tfidf_scores([]) == []


# ---------------------------------------------------------------------------
# Chunk pruner tests
# ---------------------------------------------------------------------------

class TestPruneChunks:
    SAMPLE_CHUNKS = [
        "The KV-cache stores intermediate attention states for transformer models.",
        "Byte-Pair Encoding builds a vocabulary by merging frequent character pairs.",
        "Quantization reduces model weight precision to lower bit-widths like INT4.",
        "Time-to-First-Token measures latency from prompt submission to first token.",
        "Ethiopic script encodes consonant-vowel pairs as single abugida characters.",
    ]

    def test_returns_within_budget(self):
        budget = 50  # very tight
        pruned, total = prune_chunks(self.SAMPLE_CHUNKS, budget)
        assert total <= budget + 10, (
            f"Pruned context ({total}) significantly exceeds budget ({budget})"
        )

    def test_large_budget_keeps_all(self):
        budget = 10_000  # effectively unlimited
        pruned, total = prune_chunks(self.SAMPLE_CHUNKS, budget)
        assert len(pruned) == len(self.SAMPLE_CHUNKS), (
            f"With large budget, expected all {len(self.SAMPLE_CHUNKS)} chunks, "
            f"got {len(pruned)}"
        )

    def test_output_is_subset_of_input(self):
        budget = 100
        pruned, _ = prune_chunks(self.SAMPLE_CHUNKS, budget)
        # Each pruned chunk must start with content from one of the originals
        for chunk in pruned:
            # Allow for trimming — the beginning must match an original start
            assert any(
                orig.startswith(chunk[:30]) or chunk[:30] in orig
                for orig in self.SAMPLE_CHUNKS
            ), f"Pruned chunk not traceable to input: {chunk[:60]}"

    def test_empty_chunks(self):
        pruned, total = prune_chunks([], 1000)
        assert pruned == []
        assert total == 0

    def test_zero_budget(self):
        pruned, total = prune_chunks(self.SAMPLE_CHUNKS, 0)
        assert total == 0 or len(pruned) == 0


# ---------------------------------------------------------------------------
# KV-cache memory estimation
# ---------------------------------------------------------------------------

class TestKvCacheBytes:
    def test_scales_linearly_with_tokens(self):
        """KV-cache memory should double when token count doubles."""
        mem_1k = kv_cache_bytes(1000, 32, 8, 64, 2)
        mem_2k = kv_cache_bytes(2000, 32, 8, 64, 2)
        assert math.isclose(mem_2k / mem_1k, 2.0), (
            f"Memory should double with token count: {mem_1k} vs {mem_2k}"
        )

    def test_reference_value(self):
        """
        For Phi-3.5-Mini (32 layers, 8 KV heads, head_dim=96, FP16),
        1600 tokens should be approximately 200 MB.
        2 × 32 × 8 × 96 × 1600 × 2 = 157,286,400 bytes ≈ 150 MB
        """
        mem = kv_cache_bytes(1600, 32, 8, 96, 2)
        mem_mb = mem / (1024 ** 2)
        assert 100 < mem_mb < 300, f"Expected ~150-200 MB, got {mem_mb:.1f} MB"

    def test_positive_output(self):
        assert kv_cache_bytes(100, 24, 8, 64, 2) > 0


# ---------------------------------------------------------------------------
# Quantization router tests
# ---------------------------------------------------------------------------

class TestQuantizationRouter:
    def test_high_vram_selects_q8(self):
        quant, budget = select_quantization(7000)
        assert "Q8_0" in quant or "8-bit" in quant, (
            f"Expected Q8_0 for 7 GB VRAM, got: {quant}"
        )

    def test_mid_vram_selects_q4(self):
        quant, budget = select_quantization(5000)
        assert "Q4" in quant or "4-bit" in quant, (
            f"Expected Q4_K_M for 5 GB VRAM, got: {quant}"
        )

    def test_low_vram_selects_q3(self):
        quant, budget = select_quantization(2000)
        assert "Q3" in quant or "3-bit" in quant, (
            f"Expected Q3_K for 2 GB VRAM, got: {quant}"
        )

    def test_token_budget_increases_with_compression(self):
        _, budget_high = select_quantization(7000)
        _, budget_low = select_quantization(2000)
        assert budget_low >= budget_high, (
            f"Lower VRAM tier should support more tokens via compression: "
            f"high={budget_high}, low={budget_low}"
        )

    def test_zero_vram(self):
        quant, budget = select_quantization(0)
        assert budget > 0, "Token budget must be positive even at 0 VRAM"
