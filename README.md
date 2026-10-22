<p align="center"><img width="70%" src="https://github.com/mugurbil/prix/blob/main/prix.png" alt="PriX logo" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**PriX** is a JAX/XLA framework for **Private Accelerated Transformers** via Secure Multi-Party Computation (MPC), implementing Wavelet-encoded Look-Up Tables (Haar and Biorthogonal 2.2), SHAFT constant-round Fourier-series activations and ODE softmax, `[EGK+20]` probabilistic truncation, and Fission hybrid plaintext evaluators.

> [!WARNING]
> **Research prototype — not yet suitable for protecting real secrets.**
> - The share/triple PRNG is SplitMix64 (GPU) / PCG64 (CPU), which is fast but **not a cryptographically secure PRF**; a party that learns some outputs can predict the stream. A keyed CSPRNG (e.g. Threefry/AES-CTR) is planned.
> - The default `TFP` (Trusted First Party) provider lets party 0 generate all correlated randomness, so **party 0 can learn every secret** (as in CrypTen). Use it for benchmarking only; `TTP` assumes a non-colluding dealer.
> - Security is semi-honest (honest-but-curious) only.

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

To run the head-to-head benchmark suite against the PyTorch **CrypTen** baseline
([jimouris/curl](https://github.com/jimouris/curl), the CrypTen-based reference implementation of CuRL and Fission, augmented with [andeskyl/SHAFT](https://github.com/andeskyl/SHAFT)):
```bash
# On CPU (all four approximation modes, both frameworks, then a markdown report)
JAX_PLATFORMS=cpu python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --device cpu

# On GPU (CUDA)
python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --device cuda

# Subset of modes / more repetitions
python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --modes crypten,shaft,lut --llm-repeats 5
```

Each framework is benchmarked in four approximation modes, with identical inputs and the `TFP` provider:

| Mode | Nonlinearities | Config |
|---|---|---|
| **CrypTen approximations** | CrypTen's original methods: limit exp, Newton–Raphson reciprocal / inv-sqrt, Taylor erf, sigmoid-based SiLU, CrypTen truncation | `configs/approximations.yaml` |
| **SHAFT (Fourier + ODE)** | [SHAFT (NDSS 2025)](https://github.com/andeskyl/SHAFT): 1-round trigonometric-triple Fourier-series GeLU / SiLU / Erf / Tanh / Sigmoid and constant-round Forward-Euler ODE Softmax | `configs/shaft.yaml` |
| **CuRL wavelet LUTs** | Haar / Biorthogonal-2.2 wavelet look-up tables, EGK truncation | `configs/default.yaml` |
| **Fission evaluator** | Permute-and-reveal to a plaintext evaluator party | `configs/fission.yaml` |

---

## Benchmarks: PriX (JAX) vs. CrypTen (PyTorch)

### 1. CPU Benchmarks

#### Private LLM Inference — all modes (ms)
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · SHAFT | PriX · SHAFT | CrypTen · CuRL LUTs | PriX · CuRL LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1169.60 | 793.09 | 988.35 | 459.77 | 906.21 | 468.34 | 259.57 | 95.84 |
| BertBase (1-Block, seq=16) | 2205.78 | 765.83 | 2417.35 | 610.22 | 2126.69 | 581.06 | 1565.40 | 358.95 |
| GPT2 (1-Block, seq=16) | 2743.70 | 741.31 | 2435.16 | 541.22 | 2372.77 | 522.81 | 2043.39 | 326.48 |
| Llama (1-Block, d=768, seq=16) | 2072.30 | 579.57 | 1868.44 | 496.75 | 1924.62 | 498.19 | 1563.28 | 301.83 |
| ModernBert (1-Block, seq=16) | 1617.38 | 602.25 | 1295.80 | 380.24 | 1481.51 | 423.62 | 1069.99 | 210.09 |

#### PriX speedup over CrypTen — LLM inference
| Model | CrypTen approx. | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **1.47x** | **2.15x** | **1.93x** | **2.71x** |
| BertBase (1-Block, seq=16) | **2.88x** | **3.96x** | **3.66x** | **4.36x** |
| GPT2 (1-Block, seq=16) | **3.70x** | **4.50x** | **4.54x** | **6.26x** |
| Llama (1-Block, d=768, seq=16) | **3.58x** | **3.76x** | **3.86x** | **5.18x** |
| ModernBert (1-Block, seq=16) | **2.69x** | **3.41x** | **3.50x** | **5.09x** |

#### MPC Primitives & Nonlinearities on CPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|
| Private MatMul (128x128) | 23.17 / 7.43 (**3.12x**) | 23.58 / 6.54 (**3.61x**) | 27.32 / 7.50 (**3.64x**) | 23.93 / 6.31 (**3.79x**) |
| Private Mul (128x128) | 5.54 / 4.03 (**1.38x**) | 4.69 / 2.34 (**2.01x**) | 8.24 / 6.64 (**1.24x**) | 4.79 / 3.36 (**1.42x**) |
| Comparison / ReLU (32x64) | 26.15 / 18.61 (**1.41x**) | 25.10 / 18.34 (**1.37x**) | 30.14 / 19.58 (**1.54x**) | 26.53 / 16.93 (**1.57x**) |
| Exp (32x64) | 15.72 / 14.41 (**1.09x**) | 16.34 / 10.91 (**1.50x**) | 52.30 / 25.51 (**2.05x**) | 7.75 / 1.34 (**5.78x**) |
| Reciprocal (32x64) | 76.36 / 61.07 (**1.25x**) | 69.17 / 50.94 (**1.36x**) | 35.11 / 11.48 (**3.06x**) | 6.22 / 2.11 (**2.95x**) |
| InvSqrt (32x64) | 53.73 / 34.28 (**1.57x**) | 39.98 / 29.24 (**1.37x**) | 96.14 / 16.45 (**5.84x**) | 7.20 / 1.32 (**5.44x**) |
| GeLU (32x64) | 89.22 / 74.00 (**1.21x**) | 52.07 / 31.79 (**1.64x**) | 29.60 / 8.49 (**3.49x**) | 6.55 / 1.51 (**4.34x**) |
| SiLU (32x64) | 75.68 / 57.32 (**1.32x**) | 61.81 / 31.17 (**1.98x**) | 34.97 / 6.46 (**5.41x**) | 5.38 / 1.87 (**2.88x**) |
| Softmax (4x32x32) | 310.40 / 233.34 (**1.33x**) | 254.24 / 114.91 (**2.21x**) | 297.92 / 195.87 (**1.52x**) | 10.67 / 2.08 (**5.12x**) |
| LayerNorm (4x32x64) | 59.27 / 34.58 (**1.71x**) | 54.89 / 33.22 (**1.65x**) | 42.68 / 13.58 (**3.14x**) | 12.69 / 2.59 (**4.90x**) |

---

### 2. GPU Benchmarks (NVIDIA Tesla T4, CUDA 13.0)

> [!NOTE]
> These GPU numbers were measured with the previous benchmark script (CuRL LUT and Fission modes only, before the security fixes). They will be refreshed with `benchmark_prix_vs_crypten.py` including the CrypTen-approximations mode.

#### Private LLM Inference — CuRL Wavelet LUT Mode (ms)
| Model | CrypTen (PyTorch, ms) | PriX (JAX, ms) | Speedup (CrypTen / PriX) |
|---|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1857.90 | 1469.43 | **1.26x** |
| BertBase (1-Block, seq=16) | 1678.74 | 1089.91 | **1.54x** |
| GPT2 (1-Block, seq=16) | 1614.01 | 1109.09 | **1.46x** |
| Llama (1-Block, d=768, seq=16) | 1634.82 | 1012.78 | **1.61x** |
| ModernBert (1-Block, seq=16) | 1438.15 | 954.41 | **1.51x** |

#### Private LLM Inference — Fission Evaluator Mode (ms)
| Model | CrypTen (PyTorch, ms) | PriX (JAX, ms) | Speedup (CrypTen / PriX) |
|---|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 395.51 | 152.61 | **2.59x** |
| BertBase (1-Block, seq=16) | 927.31 | 487.84 | **1.90x** |
| GPT2 (1-Block, seq=16) | 850.89 | 442.06 | **1.92x** |
| Llama (1-Block, d=768, seq=16) | 949.89 | 372.28 | **2.55x** |
| ModernBert (1-Block, seq=16) | 683.50 | 259.99 | **2.63x** |

#### MPC Primitives & CuRL Wavelet LUT Microbenchmarks on GPU (ms)
| Benchmark | CrypTen (PyTorch, ms) | PriX (JAX, ms) | Speedup (CrypTen / PriX) |
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

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [JAX](https://github.com/jax-ml/jax).
