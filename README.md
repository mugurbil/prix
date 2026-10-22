<p align="center"><img width="35%" src="https://github.com/mugurbil/prix/blob/main/prix.png" alt="PriX logo" /></p>

[![PyPI version](https://img.shields.io/pypi/v/prix-jax.svg)](https://pypi.org/project/prix-jax/)
[![Tests](https://github.com/mugurbil/prix/actions/workflows/tests.yml/badge.svg)](https://github.com/mugurbil/prix/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/prix_vs_curl_gpu_colab.ipynb)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**PriX** is a **modern, high-performance JAX/XLA drop-in alternative to [CrypTen (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html)** built for **Private Machine Learning and Private LLM / Transformer Research** via Secure Multi-Party Computation (MPC).

Write familiar JAX/PyTorch-style Python (`prix.array`, `@prix.mpc.run_multiprocess`, `prix.nn.Module`, `.reconstruct()` / `.get_plain_text()`) while running on modern **Python 3.10–3.13** and **XLA (CPU, CUDA GPU, TPU)**—and switch between four state-of-the-art MPC approximation paradigms ([**CuRL (CAMLIS 2024)**](https://eprint.iacr.org/2024/1045), [**SHAFT (NDSS 2025)**](https://eprint.iacr.org/2024/1965), [**CrypTen (NeurIPS 2021)**](https://arxiv.org/abs/2109.00984), and [**Fission (IEEE CNS 2025)**](https://eprint.iacr.org/2025/653)) with a single argument (`mode="lut" | "shaft" | "crypten" | "fission"`).

> [!WARNING]
> **Research prototype — not yet suitable for protecting real secrets.**
> - The share/triple PRNG is SplitMix64 (GPU) / PCG64 (CPU), which is fast but **not a cryptographically secure PRF**; a party that learns some outputs can predict the stream. A keyed CSPRNG (e.g. Threefry/AES-CTR) is planned.
> - The default `TFP` (Trusted First Party) provider lets party 0 generate all correlated randomness, so **party 0 can learn every secret** (as in CrypTen). Use it for benchmarking only; `TTP` assumes a non-colluding dealer.
> - Security is semi-honest (honest-but-curious) only.

---

## Why PriX over CrypTen?

Original [CrypTen (NeurIPS 2021)](https://github.com/facebookresearch/CrypTen) pioneered ML-friendly MPC in PyTorch, but has been archived since 2023 and is pinned to legacy Python/PyTorch versions, while recent MPC breakthroughs—[CuRL (CAMLIS 2024)](https://eprint.iacr.org/2024/1045), [SHAFT (NDSS 2025)](https://eprint.iacr.org/2024/1965), and [Fission (IEEE CNS 2025)](https://eprint.iacr.org/2025/653)—live in separate PyTorch forks ([jimouris/curl](https://github.com/jimouris/curl), [andeskyl/SHAFT](https://github.com/andeskyl/SHAFT)). **PriX** unifies all four in a single, actively maintained JAX/XLA package:

| Feature | CrypTen (PyTorch) | **PriX (JAX/XLA)** |
|---|---|---|
| **Python & Accelerator Support** | Legacy Python / `torch==1.9` | **Python 3.10–3.13**, modern **CPU, CUDA GPU, & TPU** via `pip install prix-jax` |
| **API Ergonomics** | `crypten.cryptensor`, `@run_multiprocess`, `crypten.nn` | **`prix.array`, `@prix.mpc.run_multiprocess`, `prix.nn`** (plus drop-in `prix.cryptensor` alias) |
| **Nonlinear Approximation Backends** | Polynomial / NR / limit only (or separate forks) | **4 built-in backends in 1 line** (`mode="lut" \| "shaft" \| "crypten" \| "fission"`) |
| **Kernel Execution** | Eager PyTorch op-by-op launches | **Whole-graph `@jax.jit` XLA fusion** (fused Kogge-Stone adder, LUT gather, Fourier & ODE steps) |
| **Performance (CPU & GPU)** | Baseline (`1.0x`) | **1.5x–6.3x faster on CPU**, **2.0x–10.3x faster on GPU** across all 60 benchmarks |

---

## Installation

Install from PyPI:
```bash
pip install prix-jax
```

Or install from source:
```bash
git clone https://github.com/mugurbil/prix.git
cd prix
pip install -e .
```

---

## 30-Second Quickstart for Researchers

### 1. Secret Sharing & Multi-Party Computation

If you know JAX, PyTorch, or CrypTen, you already know PriX. Decorate any function with `@prix.mpc.run_multiprocess` (or call `prix.run(fn, ...)`), secret-share tensors with `prix.array(...)` (or `prix.cryptensor(...)`), and reconstruct with `.reconstruct()` (or `.get_plain_text()`):

```python
import prix

@prix.mpc.run_multiprocess(world_size=2, mode="shaft")
def private_inference_demo():
    # Party 0 secret-shares input x; Party 1 secret-shares weights w
    x = prix.array([[1.0, -0.5, 2.0]], src=0)
    w = prix.array([[0.5, 1.0], [-1.0, 0.25], [0.8, -0.5]], src=1)

    # Private matrix multiplication + nonlinear activations under MPC
    logits = (x @ w).gelu()
    probs = logits.softmax(dim=-1)

    # Reconstruct plaintext result (or use .get_plain_text())
    return probs.reconstruct()

print("Decrypted probabilities:", private_inference_demo())
```

### 2. One-Line Ablations Across MPC Approximation Paradigms

Running ablation studies for a paper? Switch between all four MPC nonlinear approximation backends using `mode=` (or `prix.set_mode(...)`) without touching your model code:

```python
import prix

def eval_gelu():
    x = prix.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    return x.gelu().reconstruct()

for mode in ["crypten", "shaft", "lut", "fission"]:
    out = prix.run(eval_gelu, world_size=2, mode=mode)
    print(f"[{mode:>7}] GeLU:", out.round(4))
```

| `mode` | Nonlinearities & Truncation | Paper & Reference Code |
|---|---|---|
| **`"lut"`** *(default)* | Haar & Biorthogonal-2.2 Wavelet Look-Up Tables + `[EGK+20]` probabilistic truncation (`configs/default.yaml`) | [**CuRL (CAMLIS 2024)**](https://eprint.iacr.org/2024/1045) · [**Wave Hello (PoPETs 2025)**](https://doi.org/10.56553/popets-2025-0083) · [**Ripple (ISC 2024)**](https://eprint.iacr.org/2024/866) · [jimouris/curl](https://github.com/jimouris/curl) |
| **`"shaft"`** | 1-round trigonometric-triple Fourier-series GeLU / SiLU / Erf / Tanh / Sigmoid + constant-round Forward-Euler ODE Softmax (`configs/shaft.yaml`) | [**SHAFT (NDSS 2025)**](https://eprint.iacr.org/2024/1965) · [andeskyl/SHAFT](https://github.com/andeskyl/SHAFT) |
| **`"crypten"`** | Classic limit `exp`, Newton–Raphson `reciprocal` / `inv_sqrt`, Taylor `erf`, sigmoid-based `silu`, CrypTen truncation (`configs/approximations.yaml`) | [**CrypTen (NeurIPS 2021)**](https://arxiv.org/abs/2109.00984) · [facebookresearch/CrypTen](https://github.com/facebookresearch/CrypTen) |
| **`"fission"`** | Hybrid permute-and-reveal to a non-colluding plaintext evaluator party, auto-spawned by `prix.run` (`configs/fission.yaml`) | [**Fission (IEEE CNS 2025)**](https://eprint.iacr.org/2025/653) · [jimouris/curl](https://github.com/jimouris/curl) |

### 3. Private Neural Networks & Transformers (`prix.nn`)

Build models with `prix.nn.Module` (`Linear`, `LayerNorm`, `MultiheadAttention`, `GeLU`, `SiLU`, `Softmax`, `Sequential`, etc.), call `.encrypt()`, and run private forward passes:

```python
import prix
import prix.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, d_model=16, d_ff=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))

@prix.mpc.run_multiprocess(world_size=2, mode="lut")
def run_private_mlp():
    model = TinyMLP(d_model=16, d_ff=64).encrypt(src=0)
    x = prix.randn(2, 8, 16)  # secret-shared random input (batch=2, seq=8, d=16)
    y = model(x)
    return y.reconstruct().shape

print("Output shape:", run_private_mlp())
```

---

## Documentation & Interactive Colab Tutorials

### Interactive Google Colab Notebooks
| Notebook | Description | Colab Link |
|---|---|---|
| [**01 · PriX in 5 Minutes: MPC Basics**](examples/01_quickstart_mpc_basics.ipynb) | Secret sharing across parties (`src=0`, `src=1`), `@prix.mpc.run_multiprocess`, private arithmetic, comparisons, and `where` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/01_quickstart_mpc_basics.ipynb) |
| [**02 · Comparing MPC Nonlinearities**](examples/02_approximations_and_ablations.ipynb) | Accuracy & latency ablations across **CuRL LUTs**, **SHAFT**, **CrypTen**, and **Fission**, plus `prix.cfg` hyperparameter sweeps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/02_approximations_and_ablations.ipynb) |
| [**03 · Private Transformers & LLMs (`prix.nn`)**](examples/03_private_transformers_and_llms.ipynb) | Building custom `prix.nn.Module` Transformer blocks, loading `state_dict` weights, and encrypting models with `.encrypt()` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/03_private_transformers_and_llms.ipynb) |
| [**04 · GPU Benchmark Suite (PriX vs. CrypTen)**](examples/prix_vs_curl_gpu_colab.ipynb) | End-to-end reproducible GPU benchmark comparing PriX vs. CrypTen across all 4 modes and 5 LLM architectures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/prix_vs_curl_gpu_colab.ipynb) |

### Guides & Reference Docs
- [**PriX Technical Report / Paper (PDF)**](paper/prix_arxiv.pdf) ([LaTeX source](paper/main.tex)) — *"PriX: Compiling Private Transformers via Whole-Graph XLA Fusion in Secure Multi-Party Computation"*.
- [**Quickstart & CrypTen Migration Guide**](docs/quickstart.md) — Side-by-side CrypTen-to-PriX cheat sheet, arithmetic vs. binary secret sharing, and `TFP` vs. `TTP` providers.
- [**MPC Approximation Modes & Configuration Reference**](docs/modes_and_configuration.md) — Deep dive into `"lut"`, `"shaft"`, `"crypten"`, and `"fission"` modes and how to mix-and-match methods via `prix.cfg`.
- [**API Reference**](docs/api_reference.md) — Complete reference for top-level `prix` functions, `prix.Array` methods, and `prix.nn` neural network layers.

---

## Running Tests & Reproducing Benchmarks

To run the unit test suite on CPU:
```bash
JAX_PLATFORMS=cpu python3 -m unittest discover -s tests -q
```

To run the head-to-head benchmark suite against the PyTorch **CrypTen** baseline
([jimouris/curl](https://github.com/jimouris/curl), the CrypTen-based reference implementation of [CuRL (CAMLIS 2024)](https://eprint.iacr.org/2024/1045) and [Fission (IEEE CNS 2025)](https://eprint.iacr.org/2025/653), augmented with [SHAFT (NDSS 2025)](https://github.com/andeskyl/SHAFT)):
```bash
# On CPU (all four approximation modes, both frameworks, then a markdown report)
JAX_PLATFORMS=cpu python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --device cpu

# On GPU (CUDA)
python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --device cuda

# Subset of modes / more repetitions
python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --modes crypten,shaft,lut --llm-repeats 5
```

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
| BertTiny (2-Block, seq=16) | 1891.21 | 831.48 | 1206.41 | 497.70 | 1928.25 | 641.74 | 393.31 | 144.88 |
| BertBase (1-Block, seq=16) | 1651.63 | 753.54 | 1303.92 | 623.69 | 1680.62 | 658.69 | 908.80 | 428.51 |
| GPT2 (1-Block, seq=16) | 1648.83 | 780.20 | 1284.63 | 611.72 | 1611.53 | 645.35 | 860.02 | 436.71 |
| Llama (1-Block, d=768, seq=16) | 1596.48 | 628.52 | 1284.00 | 488.35 | 1681.80 | 554.73 | 978.78 | 354.19 |
| ModernBert (1-Block, seq=16) | 1439.01 | 552.47 | 1094.72 | 415.51 | 1438.62 | 466.32 | 705.92 | 239.02 |

#### PriX speedup over CrypTen — LLM inference on GPU
| Model | CrypTen approx. | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **2.27x** | **2.42x** | **3.00x** | **2.71x** |
| BertBase (1-Block, seq=16) | **2.19x** | **2.09x** | **2.55x** | **2.12x** |
| GPT2 (1-Block, seq=16) | **2.11x** | **2.10x** | **2.50x** | **1.97x** |
| Llama (1-Block, d=768, seq=16) | **2.54x** | **2.63x** | **3.03x** | **2.76x** |
| ModernBert (1-Block, seq=16) | **2.60x** | **2.63x** | **3.09x** | **2.95x** |

#### MPC Primitives & Nonlinearities on GPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|
| Private MatMul (128x128) | 13.32 / 3.37 (**3.96x**) | 11.29 / 5.43 (**2.08x**) | 16.16 / 7.05 (**2.29x**) | 11.36 / 3.53 (**3.21x**) |
| Private Mul (128x128) | 4.06 / 4.00 (**1.01x**) | 4.09 / 2.60 (**1.57x**) | 9.77 / 5.09 (**1.92x**) | 4.23 / 2.74 (**1.54x**) |
| Comparison / ReLU (32x64) | 45.60 / 13.77 (**3.31x**) | 44.99 / 13.09 (**3.44x**) | 51.06 / 13.67 (**3.74x**) | 50.62 / 13.79 (**3.67x**) |
| Exp (32x64) | 28.31 / 16.73 (**1.69x**) | 27.10 / 17.57 (**1.54x**) | 67.27 / 24.89 (**2.70x**) | 6.99 / 1.52 (**4.61x**) |
| Reciprocal (32x64) | 120.16 / 65.95 (**1.82x**) | 134.18 / 62.83 (**2.14x**) | 34.20 / 12.07 (**2.83x**) | 8.58 / 1.59 (**5.40x**) |
| InvSqrt (32x64) | 76.28 / 37.66 (**2.03x**) | 81.15 / 36.33 (**2.23x**) | 130.45 / 12.67 (**10.30x**) | 7.24 / 1.74 (**4.15x**) |
| GeLU (32x64) | 142.86 / 72.66 (**1.97x**) | 93.29 / 25.61 (**3.64x**) | 24.22 / 12.61 (**1.92x**) | 7.33 / 1.75 (**4.20x**) |
| SiLU (32x64) | 124.01 / 46.51 (**2.67x**) | 92.20 / 27.90 (**3.30x**) | 25.96 / 12.99 (**2.00x**) | 7.58 / 1.60 (**4.74x**) |
| Softmax (4x32x32) | 543.45 / 221.37 (**2.45x**) | 225.10 / 93.67 (**2.40x**) | 491.56 / 190.68 (**2.58x**) | 7.40 / 1.59 (**4.65x**) |
| LayerNorm (4x32x64) | 92.08 / 42.26 (**2.18x**) | 96.84 / 45.36 (**2.14x**) | 149.84 / 24.11 (**6.22x**) | 9.34 / 3.40 (**2.75x**) |

---

## References

PriX implements and accelerates the protocols and nonlinear approximations introduced in the following works:

- **CrypTen (`mode="crypten"`)** — Brian Knott, Shobha Venkataraman, Awni Hannun, Shubho Sengupta, Mark Ibrahim, and Laurens van der Maaten. *"CrypTen: Secure Multi-Party Computation Meets Machine Learning."* **NeurIPS 2021** ([Paper](https://arxiv.org/abs/2109.00984) · [Code](https://github.com/facebookresearch/CrypTen)).
- **CuRL (`mode="lut"`)** — Manuel B. Santos, Dimitris Mouris, Mehmet Ugurbil, Stanislaw Jarecki, José Reis, Shubho Sengupta, and Miguel de Vega. *"Curl: Private LLMs through Wavelet-Encoded Look-Up Tables."* **CAMLIS 2024** ([ePrint 2024/1045](https://eprint.iacr.org/2024/1045) · [Code](https://github.com/jimouris/curl)).
- **Wave Hello to Privacy (`mode="lut"`)** — José Reis, Mehmet Ugurbil, Sameer Wagh, Ryan Henry, and Miguel de Vega. *"Wave Hello to Privacy: Efficient Mixed-Mode MPC using Wavelet Transforms."* **PoPETs 2025** ([Paper](https://doi.org/10.56553/popets-2025-0083)).
- **Ripple (`mode="lut"`)** — Charles Gouert, Mehmet Ugurbil, Dimitris Mouris, Miguel de Vega, and Nektarios G. Tsoutsos. *"Ripple: Accelerating Programmable Bootstraps for FHE with Wavelet Approximations."* **ISC 2024** ([ePrint 2024/866](https://eprint.iacr.org/2024/866)).
- **SHAFT (`mode="shaft"`)** — Andes Y. L. Kei and Sherman S. M. Chow. *"SHAFT: Secure, Handy, Accurate, and Fast Transformer Inference."* **NDSS 2025** ([ePrint 2024/1965](https://eprint.iacr.org/2024/1965) · [Code](https://github.com/andeskyl/SHAFT)).
- **Fission (`mode="fission"`)** — Mehmet Ugurbil, Dimitris Mouris, Manuel B. Santos, José Cabrero-Holgueras, Miguel de Vega, and Shubho Sengupta. *"Fission: Distributed Privacy-Preserving Large Language Model Inference."* **IEEE CNS 2025** ([ePrint 2025/653](https://eprint.iacr.org/2025/653) · [Code](https://github.com/jimouris/curl)).

### Citing PriX

If you use **PriX** in your research, please cite our technical report ([`paper/prix_arxiv.pdf`](paper/prix_arxiv.pdf)):

```bibtex
@article{ugurbil2026prix,
  author  = {Mehmet Ugurbil},
  title   = {{PriX}: Compiling Private Transformers via Whole-Graph {XLA} Fusion in Secure Multi-Party Computation},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/mugurbil/prix}
}
```

---

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [JAX](https://github.com/jax-ml/jax).
