<p align="center"><img width="70%" src="https://github.com/mugurbil/prix/blob/main/prix.png" alt="PriX logo" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**PriX** is a JAX/XLA framework for **Private Accelerated Transformers** via Secure Multi-Party Computation (MPC), implementing Wavelet-encoded Look-Up Tables (Haar and Biorthogonal 2.2), `[EGK+20]` probabilistic truncation, and Fission hybrid plaintext evaluators.

## Installation & Quickstart

Install from PyPI:
```bash
pip install prix-jax
```

Or install from source:
```bash
pip install -e .
```

To run the unit test suite on CPU:
```bash
JAX_PLATFORMS=cpu python3 -m unittest discover -s tests -q
```

To run the head-to-head benchmark suite against [CuRL](https://github.com/jimouris/curl):
```bash
# On CPU
JAX_PLATFORMS=cpu python3 benchmarks/benchmark_prix_vs_curl.py --curl-root /path/to/curl --device cpu

# On GPU (CUDA)
python3 benchmarks/benchmark_prix_vs_curl.py --curl-root /path/to/curl --device cuda
```

---

## Benchmarks: PriX (JAX) vs. CuRL (PyTorch/CrypTen)

### 1. GPU Benchmarks (NVIDIA Tesla T4, CUDA 13.0)

#### Private LLM Inference — Wavelet LUT Mode (ms)
| Model | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |
|---|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1857.90 | 1469.43 | **1.26x** |
| BertBase (1-Block, seq=16) | 1678.74 | 1089.91 | **1.54x** |
| GPT2 (1-Block, seq=16) | 1614.01 | 1109.09 | **1.46x** |
| Llama (1-Block, d=768, seq=16) | 1634.82 | 1012.78 | **1.61x** |
| ModernBert (1-Block, seq=16) | 1438.15 | 954.41 | **1.51x** |

#### Private LLM Inference — Fission Evaluator Mode (ms)
| Model | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |
|---|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 395.51 | 152.61 | **2.59x** |
| BertBase (1-Block, seq=16) | 927.31 | 487.84 | **1.90x** |
| GPT2 (1-Block, seq=16) | 850.89 | 442.06 | **1.92x** |
| Llama (1-Block, d=768, seq=16) | 949.89 | 372.28 | **2.55x** |
| ModernBert (1-Block, seq=16) | 683.50 | 259.99 | **2.63x** |

#### MPC Primitives & Wavelet LUT Microbenchmarks on GPU (ms)
| Benchmark | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |
|---|---:|---:|---:|
| Private MatMul (128x128) | 17.92 | 8.82 | **2.03x** |
| Private Mul (128x128) | 10.17 | 7.77 | **1.31x** |
| EGK TruncPr (128x128) | 6.56 | 3.29 | **2.00x** |
| Comparison / ReLU (32x64) | 46.54 | 70.80 | **0.66x** |
| Exp [Haar LUT] (32x64) | 65.44 | 85.70 | **0.76x** |
| Reciprocal [Haar LUT] (32x64) | 34.99 | 14.90 | **2.35x** |
| InvSqrt [Haar LUT] (32x64) | 126.52 | 14.61 | **8.66x** |
| GeLU [Haar LUT] (32x64) | 23.17 | 15.46 | **1.50x** |
| SiLU [Haar LUT] (32x64) | 23.92 | 16.58 | **1.44x** |
| Softmax [LUT] (4x32x32) | 480.23 | 657.41 | **0.73x** |
| LayerNorm [LUT] (4x32x64) | 145.50 | 31.51 | **4.62x** |

---

### 2. CPU Benchmarks

#### Private LLM Inference — Wavelet LUT Mode (ms)
| Model | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |
|---|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 991.20 | 548.53 | **1.81x** |
| BertBase (1-Block, seq=16) | 2402.88 | 677.29 | **3.55x** |
| GPT2 (1-Block, seq=16) | 2344.71 | 575.77 | **4.07x** |
| Llama (1-Block, d=768, seq=16) | 1797.25 | 578.70 | **3.11x** |
| ModernBert (1-Block, seq=16) | 1308.25 | 438.92 | **2.98x** |

#### Private LLM Inference — Fission Evaluator Mode (ms)
| Model | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |
|---|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 253.73 | 118.94 | **2.13x** |
| BertBase (1-Block, seq=16) | 1594.26 | 398.84 | **4.00x** |
| GPT2 (1-Block, seq=16) | 2142.77 | 379.05 | **5.65x** |
| Llama (1-Block, d=768, seq=16) | 1573.41 | 326.26 | **4.82x** |
| ModernBert (1-Block, seq=16) | 1048.27 | 239.21 | **4.38x** |

#### MPC Primitives & Wavelet LUT Microbenchmarks on CPU (ms)
| Benchmark | CuRL (PyTorch, ms) | PriX (JAX, ms) | Speedup (CuRL / PriX) |
|---|---:|---:|---:|
| Private MatMul (128x128) | 28.12 | 7.90 | **3.56x** |
| Private Mul (128x128) | 8.35 | 7.20 | **1.16x** |
| EGK TruncPr (128x128) | 3.94 | 3.63 | **1.08x** |
| Comparison / ReLU (32x64) | 26.90 | 19.22 | **1.40x** |
| Exp [Haar LUT] (32x64) | 62.06 | 27.29 | **2.27x** |
| Reciprocal [Haar LUT] (32x64) | 41.72 | 10.56 | **3.95x** |
| InvSqrt [Haar LUT] (32x64) | 124.84 | 18.90 | **6.61x** |
| GeLU [Haar LUT] (32x64) | 26.80 | 6.10 | **4.39x** |
| SiLU [Haar LUT] (32x64) | 32.54 | 9.01 | **3.61x** |
| Softmax [LUT] (4x32x32) | 276.64 | 205.43 | **1.35x** |
| LayerNorm [LUT] (4x32x64) | 44.89 | 16.94 | **2.65x** |

---

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [JAX](https://github.com/jax-ml/jax).
