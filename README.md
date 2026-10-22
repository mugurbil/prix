<p align="center"><img width="35%" src="https://github.com/mugurbil/prix/blob/main/prix.png" alt="PriX logo" /></p>

[![PyPI version](https://img.shields.io/pypi/v/prix-jax.svg)](https://pypi.org/project/prix-jax/)
[![Tests](https://github.com/mugurbil/prix/actions/workflows/tests.yml/badge.svg)](https://github.com/mugurbil/prix/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/prix_vs_curl_gpu_colab.ipynb)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**PriX** is a **modern, high-performance JAX/XLA drop-in alternative to [CrypTen (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html)** built for **Private Machine Learning and Private LLM / Transformer Research** via Secure Multi-Party Computation (MPC).

Write familiar JAX/PyTorch-style Python (`prix.array`, `@prix.mpc.run_multiprocess`, `prix.nn.Module`, `.reconstruct()` / `.get_plain_text()`) while running on modern **Python 3.10–3.13** and **XLA (CPU, CUDA GPU, TPU)**—and switch between five state-of-the-art MPC approximation paradigms ([**CuRL (CAMLIS 2024)**](https://eprint.iacr.org/2024/1045), [**PUMA (2023)**](https://arxiv.org/abs/2307.12533) / [**Nimbus (NeurIPS 2024)**](https://arxiv.org/abs/2411.15707), [**SHAFT (NDSS 2025)**](https://eprint.iacr.org/2024/1965), [**CrypTen (NeurIPS 2021)**](https://arxiv.org/abs/2109.00984), and [**Fission (IEEE CNS 2025)**](https://eprint.iacr.org/2025/653)) with a single argument (`mode="lut" | "poly" | "shaft" | "crypten" | "fission"`).

> [!WARNING]
> **Research prototype — not yet suitable for protecting real secrets.**
> - The share/triple PRNG is SplitMix64 (GPU) / PCG64 (CPU), which is fast but **not a cryptographically secure PRF**; a party that learns some outputs can predict the stream. A keyed CSPRNG (e.g. Threefry/AES-CTR) is planned.
> - The default `TFP` (Trusted First Party) provider lets party 0 generate all correlated randomness, so **party 0 can learn every secret** (as in CrypTen). Use it for benchmarking only; `TTP` assumes a non-colluding dealer.
> - Security is semi-honest (honest-but-curious) only.

---

## Why PriX over CrypTen?

Original [CrypTen (NeurIPS 2021)](https://github.com/facebookresearch/CrypTen) pioneered ML-friendly MPC in PyTorch, but has been archived since 2023 and is pinned to legacy Python/PyTorch versions, while recent MPC breakthroughs—[CuRL (CAMLIS 2024)](https://eprint.iacr.org/2024/1045), [PUMA (2023)](https://arxiv.org/abs/2307.12533) / [Nimbus (NeurIPS 2024)](https://arxiv.org/abs/2411.15707), [SHAFT (NDSS 2025)](https://eprint.iacr.org/2024/1965), and [Fission (IEEE CNS 2025)](https://eprint.iacr.org/2025/653)—live in separate repositories ([jimouris/curl](https://github.com/jimouris/curl), [secretflow/spu](https://github.com/secretflow/spu), [andeskyl/SHAFT](https://github.com/andeskyl/SHAFT)). **PriX** unifies all five in a single, actively maintained JAX/XLA package:

| Feature | CrypTen (PyTorch) | **PriX (JAX/XLA)** |
|---|---|---|
| **Python & Accelerator Support** | Legacy Python / `torch==1.9` | **Python 3.10–3.13**, modern **CPU, CUDA GPU, & TPU** via `pip install prix-jax` |
| **API Ergonomics** | `crypten.cryptensor`, `@run_multiprocess`, `crypten.nn` | **`prix.array`, `@prix.mpc.run_multiprocess`, `prix.nn`** (plus drop-in `prix.cryptensor` alias) |
| **Nonlinear Approximation Backends** | Polynomial / NR / limit only (or separate forks) | **5 built-in backends in 1 line** (`mode="lut" \| "poly" \| "shaft" \| "crypten" \| "fission"`) |
| **Kernel Execution** | Eager PyTorch op-by-op launches | **`@jax.jit`-compiled local protocol steps** (fused carry-lookahead rounds, LUT gather, limb matmul, Fourier combination) |
| **Performance (single host, CPU & GPU)** | Baseline (`1.0x`) | Lower latency in every configuration measured. LLM blocks: **2.8x–9.2x faster on CPU**, **2.0x–3.1x faster on GPU**. Primitives: 1.8x–20.8x on CPU, 1.0x–10.3x on GPU ([details](#benchmarks-prix-jax-vs-crypten-pytorch)) |

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

Running ablation studies for a paper? Switch between all five MPC nonlinear approximation backends using `mode=` (or `prix.set_mode(...)`) without touching your model code:

```python
import prix

def eval_gelu():
    x = prix.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    return x.gelu().reconstruct()

for mode in ["crypten", "poly", "shaft", "lut", "fission"]:
    out = prix.run(eval_gelu, world_size=2, mode=mode)
    print(f"[{mode:>7}] GeLU:", out.round(4))
```

| `mode` | Nonlinearities & Truncation | Paper & Reference Code |
|---|---|---|
| **`"lut"`** *(default)* | Haar & Biorthogonal-2.2 Wavelet Look-Up Tables + `[EGK+20]` probabilistic truncation (`configs/default.yaml`) | [**CuRL (CAMLIS 2024)**](https://eprint.iacr.org/2024/1045) · [jimouris/curl](https://github.com/jimouris/curl) |
| **`"poly"`** | Piecewise low-degree polynomials for `gelu`, `silu`, `exp`, `tanh`, `sigmoid`, `erf` (1 batched comparison + degree-4 polynomial, no LUTs, `configs/poly.yaml`) | [**PUMA (arXiv 2023)**](https://arxiv.org/abs/2307.12533) · [**Nimbus (NeurIPS 2024)**](https://arxiv.org/abs/2411.15707) · [secretflow/spu](https://github.com/secretflow/spu) |
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
| [**02 · Comparing MPC Nonlinearities**](examples/02_approximations_and_ablations.ipynb) | Accuracy & latency ablations across **CuRL LUTs**, **PUMA/Nimbus Poly**, **SHAFT**, **CrypTen**, and **Fission**, plus `prix.cfg` hyperparameter sweeps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/02_approximations_and_ablations.ipynb) |
| [**03 · Private Transformers & LLMs (`prix.nn`)**](examples/03_private_transformers_and_llms.ipynb) | Building custom `prix.nn.Module` Transformer blocks, loading `state_dict` weights, and encrypting models with `.encrypt()` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/03_private_transformers_and_llms.ipynb) |
| [**04 · GPU Benchmark Suite (PriX vs. CrypTen)**](examples/prix_vs_curl_gpu_colab.ipynb) | End-to-end reproducible GPU benchmark comparing PriX vs. CrypTen across all 5 modes and 5 LLM architectures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/prix_vs_curl_gpu_colab.ipynb) |

### Guides & Reference Docs
- [**PriX Technical Report / Paper (PDF)**](paper/prix_arxiv.pdf) ([LaTeX source](paper/main.tex)) — *"PriX: Private Accelerated Transformers via Secure Multi-Party Computation in JAX"*.
- [**Quickstart & CrypTen Migration Guide**](docs/quickstart.md) — Side-by-side CrypTen-to-PriX cheat sheet, arithmetic vs. binary secret sharing, and `TFP` vs. `TTP` providers.
- [**MPC Approximation Modes & Configuration Reference**](docs/modes_and_configuration.md) — Deep dive into `"lut"`, `"poly"`, `"shaft"`, `"crypten"`, and `"fission"` modes and how to mix-and-match methods via `prix.cfg`.
- [**API Reference**](docs/api_reference.md) — Complete reference for top-level `prix` functions, `prix.Array` methods, and `prix.nn` neural network layers.

---

## Running Tests & Reproducing Benchmarks

To run the unit test suite on CPU:
```bash
JAX_PLATFORMS=cpu python3 -m unittest discover -s tests -q
```

To run the head-to-head benchmark suite against the PyTorch **CrypTen** baseline
([jimouris/curl](https://github.com/jimouris/curl), the CrypTen-based reference implementation of [CuRL (CAMLIS 2024)](https://eprint.iacr.org/2024/1045) and [Fission (IEEE CNS 2025)](https://eprint.iacr.org/2025/653), augmented with [PUMA (2023)](https://arxiv.org/abs/2307.12533) / [Nimbus (NeurIPS 2024)](https://arxiv.org/abs/2411.15707) and [SHAFT (NDSS 2025)](https://github.com/andeskyl/SHAFT)):
```bash
# On CPU (all five approximation modes, both frameworks, then a markdown report)
JAX_PLATFORMS=cpu python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --device cpu

# On GPU (CUDA)
python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --device cuda

# Subset of modes / more repetitions
python3 benchmarks/benchmark_prix_vs_crypten.py --crypten-root /path/to/curl --modes crypten,poly,shaft,lut --llm-repeats 5
```

---

## Benchmarks: PriX (JAX) vs. CrypTen (PyTorch)

> [!NOTE]
> **How to read these numbers.** Both frameworks run all parties as processes on **one host**. There is no network emulation, and communication volume isn't reported. CrypTen uses `gloo` over loopback TCP, and PriX uses a shared-memory ring buffer, so part of the speedup comes from the transport rather than from kernel fusion. Each primitive is the median of 5 timed runs (after 2 warmups), and each LLM block is the median of 3 timed runs (after 1 warmup). Differences of roughly 10% or less are within noise. Benchmarks measure latency only, not model accuracy. The `poly` and SHAFT baselines on the CrypTen side are one-to-one ports of PriX's algorithms; see [`benchmarks/benchmark_prix_vs_crypten.py`](benchmarks/benchmark_prix_vs_crypten.py).

### 1. CPU Benchmarks

AMD EPYC 7B12 (24 vCPUs), Python 3.13, JAX/jaxlib 0.12.0, PyTorch 2.14.1 (CPU). All five modes were measured in one session.

#### Private LLM Inference — all modes (ms)
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · Piecewise poly | PriX · Piecewise poly | CrypTen · SHAFT | PriX · SHAFT | CrypTen · CuRL LUTs | PriX · CuRL LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1116.92 | 394.21 | 1214.06 | 332.66 | 769.78 | 255.53 | 884.35 | 298.84 | 426.03 | 97.29 |
| BertBase (1-Block, seq=16) | 2111.69 | 555.14 | 2422.60 | 512.64 | 1992.19 | 464.97 | 1927.60 | 453.40 | 2067.63 | 334.52 |
| GPT2 (1-Block, seq=16) | 2725.99 | 547.24 | 2797.50 | 450.43 | 2483.98 | 424.27 | 3366.24 | 410.94 | 2690.23 | 292.49 |
| Llama (1-Block, d=768, seq=16) | 2015.02 | 407.47 | 2082.81 | 390.03 | 2015.52 | 345.54 | 2558.86 | 385.23 | 2058.74 | 268.46 |
| ModernBert (1-Block, seq=16) | 1520.72 | 386.44 | 1524.23 | 346.73 | 1536.32 | 273.18 | 2089.26 | 313.73 | 1378.75 | 200.17 |

#### PriX speedup over CrypTen — LLM inference
| Model | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **2.83x** | **3.65x** | **3.01x** | **2.96x** | **4.38x** |
| BertBase (1-Block, seq=16) | **3.80x** | **4.73x** | **4.28x** | **4.25x** | **6.18x** |
| GPT2 (1-Block, seq=16) | **4.98x** | **6.21x** | **5.85x** | **8.19x** | **9.20x** |
| Llama (1-Block, d=768, seq=16) | **4.95x** | **5.34x** | **5.83x** | **6.64x** | **7.67x** |
| ModernBert (1-Block, seq=16) | **3.94x** | **4.40x** | **5.62x** | **6.66x** | **6.89x** |

#### MPC Primitives & Nonlinearities on CPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | CuRL LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| Private MatMul (128x128) | 23.52 / 4.91 (**4.79x**) | 23.13 / 5.97 (**3.87x**) | 22.46 / 4.81 (**4.67x**) | 26.46 / 6.56 (**4.03x**) | 34.75 / 4.59 (**7.57x**) |
| Private Mul (128x128) | 4.64 / 2.63 (**1.77x**) | 4.64 / 1.84 (**2.53x**) | 5.02 / 1.89 (**2.65x**) | 8.62 / 4.67 (**1.85x**) | 6.95 / 2.77 (**2.51x**) |
| Comparison / ReLU (32x64) | 22.83 / 4.68 (**4.88x**) | 23.85 / 4.61 (**5.17x**) | 26.78 / 4.72 (**5.67x**) | 24.64 / 5.72 (**4.31x**) | 53.00 / 4.73 (**11.21x**) |
| Exp (32x64) | 15.21 / 8.14 (**1.87x**) | 43.27 / 11.50 (**3.76x**) | 14.86 / 6.33 (**2.35x**) | 48.23 / 14.16 (**3.41x**) | 12.62 / 1.11 (**11.40x**) |
| Reciprocal (32x64) | 80.84 / 33.62 (**2.40x**) | 85.78 / 29.17 (**2.94x**) | 73.62 / 32.70 (**2.25x**) | 33.38 / 6.83 (**4.89x**) | 21.93 / 1.05 (**20.80x**) |
| InvSqrt (32x64) | 46.97 / 17.91 (**2.62x**) | 50.21 / 22.03 (**2.28x**) | 53.72 / 19.71 (**2.73x**) | 97.08 / 15.35 (**6.33x**) | 22.03 / 1.26 (**17.52x**) |
| GeLU (32x64) | 89.70 / 40.91 (**2.19x**) | 43.82 / 16.38 (**2.68x**) | 59.12 / 13.74 (**4.30x**) | 21.39 / 5.90 (**3.63x**) | 12.54 / 1.15 (**10.89x**) |
| SiLU (32x64) | 71.08 / 24.27 (**2.93x**) | 49.93 / 15.69 (**3.18x**) | 63.15 / 13.41 (**4.71x**) | 27.30 / 5.16 (**5.29x**) | 11.83 / 1.20 (**9.86x**) |
| Softmax (4x32x32) | 293.28 / 103.93 (**2.82x**) | 333.74 / 100.37 (**3.33x**) | 136.18 / 55.29 (**2.46x**) | 284.79 / 82.01 (**3.47x**) | 14.64 / 1.58 (**9.28x**) |
| LayerNorm (4x32x64) | 55.72 / 22.91 (**2.43x**) | 54.56 / 18.13 (**3.01x**) | 56.83 / 17.13 (**3.32x**) | 50.91 / 14.32 (**3.55x**) | 16.36 / 2.89 (**5.66x**) |

> [!NOTE]
> In this session CrypTen's `fission`-mode numbers are inflated even for ReLU, whose protocol is identical in every mode: 53 ms there versus 23–27 ms in the other modes. The extra evaluator process competes for the same CPU, so treat the very large `fission` primitive speedups with caution.

---

### 2. GPU Benchmarks (NVIDIA Tesla T4, CUDA)

Google Colab, one Tesla T4. Four modes were measured in one session. `poly`-mode GPU results are pending a rerun with the corrected baseline.

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
- **PUMA (`mode="poly"`)** — Ye Dong, Wen-jie Lu, Yancheng Zheng, Haoqi Wu, Derun Zhao, Jin Tan, Zhicong Huang, Cheng Hong, Tao Wei, and Wenguang Chen. *"PUMA: Secure Inference of LLaMA-7B in Five Minutes."* **arXiv 2023** ([arXiv:2307.12533](https://arxiv.org/abs/2307.12533) · [Code](https://github.com/secretflow/spu)).
- **Nimbus (`mode="poly"`)** — Zhengyi Li, Kang Yang, Jin Tan, Wen-jie Lu, Haoqi Wu, Xiao Wang, Yu Yu, Derun Zhao, Yancheng Zheng, Minyi Guo, and Jingwen Leng. *"Nimbus: Secure and Efficient Two-Party Inference for Transformers."* **NeurIPS 2024** ([arXiv:2411.15707](https://arxiv.org/abs/2411.15707)).
- **SHAFT (`mode="shaft"`)** — Andes Y. L. Kei and Sherman S. M. Chow. *"SHAFT: Secure, Handy, Accurate, and Fast Transformer Inference."* **NDSS 2025** ([ePrint 2024/1965](https://eprint.iacr.org/2024/1965) · [Code](https://github.com/andeskyl/SHAFT)).
- **Fission (`mode="fission"`)** — Mehmet Ugurbil, Dimitris Mouris, Manuel B. Santos, José Cabrero-Holgueras, Miguel de Vega, and Shubho Sengupta. *"Fission: Distributed Privacy-Preserving Large Language Model Inference."* **IEEE CNS 2025** ([ePrint 2025/653](https://eprint.iacr.org/2025/653) · [Code](https://github.com/jimouris/curl)).

### Citing PriX

If you use **PriX** in your research, please cite our technical report ([`paper/prix_arxiv.pdf`](paper/prix_arxiv.pdf)):

```bibtex
@article{ugurbil2026prix,
  author  = {Mehmet Ugurbil},
  title   = {{PriX}: Private Accelerated Transformers via Secure Multi-Party Computation in {JAX}},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/mugurbil/prix}
}
```

---

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [JAX](https://github.com/jax-ml/jax).
