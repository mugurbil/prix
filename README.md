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

### 2. GPU Benchmarks (NVIDIA Tesla T4, CUDA)

#### Private LLM Inference — all modes (ms)
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · SHAFT | PriX · SHAFT | CrypTen · CuRL LUTs | PriX · CuRL LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1907.95 | 2279.82 | 1222.84 | 1319.37 | 1933.81 | 1541.65 | 423.29 | 157.57 |
| BertBase (1-Block, seq=16) | 1675.36 | 1495.02 | 1326.25 | 1024.81 | 1690.22 | 1149.24 | 907.64 | 465.88 |
| GPT2 (1-Block, seq=16) | 1612.91 | 1512.46 | 1286.20 | 1023.98 | 1638.29 | 1130.92 | 849.83 | 457.31 |
| Llama (1-Block, d=768, seq=16) | 1582.04 | 1324.62 | 1280.75 | 890.74 | 1684.98 | 1039.00 | 1000.85 | 369.72 |
| ModernBert (1-Block, seq=16) | 1428.77 | 1266.74 | 1104.81 | 818.48 | 1460.58 | 938.75 | 702.47 | 247.29 |

#### PriX speedup over CrypTen — LLM inference on GPU
| Model | CrypTen approx. | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **0.84x** | **0.93x** | **1.25x** | **2.69x** |
| BertBase (1-Block, seq=16) | **1.12x** | **1.29x** | **1.47x** | **1.95x** |
| GPT2 (1-Block, seq=16) | **1.07x** | **1.26x** | **1.45x** | **1.86x** |
| Llama (1-Block, d=768, seq=16) | **1.19x** | **1.44x** | **1.62x** | **2.71x** |
| ModernBert (1-Block, seq=16) | **1.13x** | **1.35x** | **1.56x** | **2.84x** |

#### MPC Primitives & Nonlinearities on GPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|
| Private MatMul (128x128) | 14.67 / 7.41 (**1.98x**) | 16.23 / 6.27 (**2.59x**) | 18.51 / 10.38 (**1.78x**) | 11.27 / 5.44 (**2.07x**) |
| Private Mul (128x128) | 6.60 / 6.09 (**1.08x**) | 7.58 / 5.36 (**1.41x**) | 9.79 / 7.86 (**1.25x**) | 3.89 / 4.85 (**0.80x**) |
| Comparison / ReLU (32x64) | 68.94 / 75.69 (**0.91x**) | 54.01 / 74.29 (**0.73x**) | 43.81 / 73.03 (**0.60x**) | 48.53 / 71.58 (**0.68x**) |
| Exp (32x64) | 44.81 / 32.99 (**1.36x**) | 27.74 / 34.05 (**0.81x**) | 62.76 / 89.98 (**0.70x**) | 7.11 / 1.36 (**5.23x**) |
| Reciprocal (32x64) | 138.81 / 135.23 (**1.03x**) | 122.65 / 141.75 (**0.87x**) | 32.26 / 15.01 (**2.15x**) | 7.75 / 1.41 (**5.50x**) |
| InvSqrt (32x64) | 83.08 / 91.92 (**0.90x**) | 79.91 / 88.38 (**0.90x**) | 126.92 / 15.39 (**8.25x**) | 7.64 / 1.39 (**5.50x**) |
| GeLU (32x64) | 157.38 / 171.37 (**0.92x**) | 92.32 / 98.21 (**0.94x**) | 24.16 / 16.75 (**1.44x**) | 8.55 / 1.70 (**5.03x**) |
| SiLU (32x64) | 121.28 / 157.02 (**0.77x**) | 90.71 / 94.83 (**0.96x**) | 23.52 / 15.86 (**1.48x**) | 7.62 / 1.37 (**5.56x**) |
| Softmax (4x32x32) | 562.02 / 720.44 (**0.78x**) | 210.82 / 268.87 (**0.78x**) | 506.30 / 683.60 (**0.74x**) | 7.85 / 1.50 (**5.23x**) |
| LayerNorm (4x32x64) | 99.04 / 98.17 (**1.01x**) | 98.10 / 100.70 (**0.97x**) | 151.43 / 33.47 (**4.52x**) | 9.06 / 3.25 (**2.79x**) |

---

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [JAX](https://github.com/jax-ml/jax).
