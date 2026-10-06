# Ge'ez-EdgeRAG: Mitigating Subword Tokenization Overhead and KV-Cache Exhaustion in Quantized RAG for Low-Resource Scripts on Consumer GPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware: NVIDIA Blackwell RTX 5060](https://img.shields.io/badge/GPU-RTX%205060%20(8GB)-76B900.svg)](https://developer.nvidia.com/)
[![Status: Research Benchmark](https://img.shields.io/badge/Status-Active%20Benchmark-success.svg)]()

**Ge'ez-EdgeRAG** is a lightweight telemetry profiler and dynamic context-pruning middleware designed for self-hosted Retrieval-Augmented Generation (RAG) pipelines operating under strict consumer GPU memory ceilings (8 GB VRAM).

---

## Statement of Need

Deploying cloud-based Large Language Models (LLMs) for regional enterprise and educational applications in low-resource settings introduces prohibitive token costs and network latency. While consumer accelerators—such as the NVIDIA GeForce RTX 5060 (8 GB GDDR7)—enable local inference of quantized Small Language Models (SLMs), standard Byte-Pair Encoding (BPE) tokenizers exhibit severe **subword fertility degradation** on non-Latin scripts such as **Ge'ez (Amharic / Tigrinya)**.

Because standard vocabularies under-represent Ethiopic characters, tokenizers frequently fall back to UTF-8 byte-level splitting. Consequently:

1. **Token Inflation (Fertility Tax):** Equivalent semantic content in Amharic consumes **4.2x to 6.5x more tokens** than English.
2. **KV-Cache Exhaustion:** A standard 5-chunk RAG context window (~1,600 tokens in English) expands to **7,000–10,000+ tokens** in Ge'ez script, rapidly exhausting the 8 GB VRAM ceiling and triggering Out-Of-Memory (OOM) crashes or severe thermal throttling.
3. **Time-To-First-Token (TTFT) Spikes:** Prefill latency scales quadratically with unpruned byte-fallback sequences.

`Ge'ez-EdgeRAG` solves this bottleneck **before** the prompt reaches the GPU by profiling live NVML hardware telemetry, calculating token fertility overhead, and dynamically pruning low-density retrieval chunks to guarantee stable, bounded-memory inference.

---

## System Architecture

```
+-----------------------------------------------------------------------+
|                        Incoming Bilingual Query                       |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|               Vector / SQL Document Retrieval (Top-K Chunks)          |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+=======================================================================+
|                     Ge'ez-EdgeRAG Middleware Layer                    |
|                                                                       |
|  1. Live NVML Telemetry Check   --> Queries Free VRAM on RTX 5060     |
|  2. Fertility Ratio Estimator   --> Detects Ge'ez Byte-Fallback Rate  |
|  3. TF-IDF / Extractive Pruner  --> Strips Low-Entropy Sentences      |
|  4. Dynamic Quantization Router --> Allocates Safe KV-Cache Budget    |
+===================================+===================================+
                                    |
                                    v
+-----------------------------------------------------------------------+
|          Local Quantized SLM Inference (Q8_0 / Q4_K_M / Q3_K)         |
|                   NVIDIA RTX 5060 (8 GB VRAM Ceiling)                 |
+-----------------------------------------------------------------------+
```

---

## Repository Structure

```
geez-edgerag-bench/
├── step1_token_tax.py       # Intrinsic tokenizer fertility & KV-cache memory profiler
├── step2_middleware.py      # VRAM-aware Ge'ez context compression & pruning engine
├── step3_benchmark.py       # End-to-end latency (TTFT, TPS) & NVML telemetry suite
├── data/
│   └── parallel_corpus.json # Parallel English-Amharic technical RAG evaluation chunks
├── results/
│   └── .gitkeep             # Benchmark outputs are written here (auto-generated)
├── tests/
│   └── test_middleware.py   # Automated unit tests for JOSS/SoftwareX reproducibility
├── requirements.txt         # Pinned Python dependencies
├── LICENSE                  # OSI-approved MIT License
└── README.md                # Project documentation
```

---

## Installation

### Prerequisites

- **OS:** Windows 11, Ubuntu 22.04+, or WSL2
- **Python:** 3.10 or higher
- **GPU (Recommended for Telemetry):** NVIDIA GPU with standard drivers installed
  *(tested on **NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM**). CPU fallback mode is supported for unit testing.*

### Setup

Clone the repository and initialize a virtual environment:

```bash
git clone https://github.com/Brooklss/geez-edgerag-bench.git
cd geez-edgerag-bench
python -m venv venv
```

Activate the virtual environment and install dependencies:

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS / WSL
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## Reproducing the Paper Experiments

### Experiment 1: Quantifying the Ge'ez Tokenization & KV-Cache Tax

Run the intrinsic profiler to measure token inflation across multilingual and byte-fallback BPE architectures (`Qwen-2.5`, `Phi-3.5-Mini`, and `SmolLM2`):

```bash
python step1_token_tax.py
```

**Expected Output Sample:**

| Tokenizer Architecture           | EN Tokens | AM Tokens | Fertility Tax | 5-Chunk Context (EN)  | 5-Chunk Context (AM)  |
| :-------------------------------- | :-------: | :-------: | :-----------: | :-------------------: | :-------------------: |
| Qwen-2.5 (Multilingual BPE)      | 26        | 118       | 4.54x         | 1600 tok (200.0 MB)   | 7261 tok (907.7 MB)   |
| Phi-3.5-Mini (Byte-Fallback BPE) | 29        | 164       | 5.66x         | 1600 tok (200.0 MB)   | 9048 tok (1131.0 MB)  |
| SmolLM2 (Byte-Level BPE)         | 27        | 172       | 6.37x         | 1600 tok (200.0 MB)   | 10192 tok (1274.0 MB) |

### Experiment 2: Running the VRAM-Aware Context Pruner

Execute the middleware pipeline to compress retrieved Amharic chunks dynamically:

```bash
python step2_middleware.py
```

This prints a live NVML telemetry snapshot and the pruned/compressed context ready for SLM inference.

### Experiment 3: End-to-End Benchmark Suite

Run the full telemetry benchmark (TTFT, TPS, VRAM delta, thermal throttle detection):

```bash
python step3_benchmark.py
```

Results are automatically written to `results/telemetry_logs.csv`.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover the middleware fertility estimator, chunk pruner, and KV-cache budget allocator in CPU fallback mode (no GPU required).

---

## Key Concepts

### Subword Fertility

The **fertility ratio** of a tokenizer on a given text is defined as:

```
fertility(T, s) = |tokens(s)| / |words(s)|
```

For Ge'ez script, byte-fallback BPE tokenizers produce fertility ratios of **4.2x–6.5x** compared to English on semantically equivalent content.

### KV-Cache Memory Budget

KV-cache memory consumption scales as:

```
KV_mem(t) = 2 x num_layers x num_kv_heads x head_dim x t x dtype_bytes
```

For a 3.8B parameter model (Phi-3.5-Mini) at FP16, a 9,000-token Amharic context consumes **~1.13 GB** of VRAM — compared to **200 MB** for the English equivalent.

### Dynamic Quantization Routing

The middleware selects a quantization tier based on available VRAM:

| Available VRAM | Quantization Level | Context Budget (Amharic) |
| :------------- | :----------------- | :----------------------- |
| >= 6 GB        | Q8_0 (8-bit)       | Up to 4,000 tokens       |
| 4–6 GB         | Q4_K_M (4-bit)     | Up to 6,000 tokens       |
| < 4 GB         | Q3_K (3-bit)       | Up to 8,000 tokens       |

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{geez_edgerag_bench_2026,
  title   = {Ge'ez-EdgeRAG: Mitigating Subword Tokenization Overhead and
             KV-Cache Exhaustion in Quantized RAG for Low-Resource Scripts
             on Consumer GPUs},
  author  = {Brooklss},
  year    = {2026},
  url     = {https://github.com/Brooklss/geez-edgerag-bench},
  license = {MIT}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
