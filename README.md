<p align="center"><img width="35%" src="prix.png" alt="PriX logo" /></p>

[![PyPI version](https://img.shields.io/pypi/v/prix-jax.svg)](https://pypi.org/project/prix-jax/)
[![Tests](https://github.com/mugurbil/prix/actions/workflows/tests.yml/badge.svg)](https://github.com/mugurbil/prix/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/prix_vs_curl_gpu_colab.ipynb)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**PriX** is a **modern, high-performance JAX/XLA drop-in alternative to [CrypTen (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html)** built for **Private Machine Learning and Private LLM / Transformer Research** via Secure Multi-Party Computation (MPC).

Write familiar JAX/PyTorch-style Python (`prix.array`, `@prix.mpc.run_multiprocess`, `prix.nn.Module`, `.reconstruct()` / `.get_plain_text()`) while running on modern **Python 3.10–3.13** and **XLA (tested on CPU and CUDA GPU)**—and switch between five state-of-the-art MPC approximation paradigms ([**Curl (CAMLIS 2024)**](https://eprint.iacr.org/2024/1127), [**PUMA (2023)**](https://arxiv.org/abs/2307.12533), [**SHAFT (NDSS 2025)**](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/), [**CrypTen (NeurIPS 2021)**](https://arxiv.org/abs/2109.00984), and [**Fission (IEEE CNS 2025)**](https://eprint.iacr.org/2025/653)) with a single argument (`mode="lut" | "poly" | "shaft" | "crypten" | "fission"`).

> [!WARNING]
> **Research prototype — not yet suitable for protecting real secrets.**
> - The share/triple PRNG is SplitMix64 (GPU) / PCG64 (CPU), which is fast but **not a cryptographically secure PRF**; a party that learns some outputs can predict the stream. A keyed CSPRNG (e.g. Threefry/AES-CTR) is planned.
> - The default `TFP` (Trusted First Party) provider lets party 0 generate all correlated randomness, so **party 0 can learn every secret** (as in CrypTen). Use it for benchmarking only; `TTP` assumes a non-colluding dealer.
> - Security is semi-honest (honest-but-curious) only.
> - Parties talk over direct peer-to-peer TCP (`prix.run`'s default; the same code runs across hosts). Enable mutual TLS for anything beyond experiments; see [Transports](docs/transports.md).

---

## Why PriX over CrypTen?

Original [CrypTen (NeurIPS 2021)](https://github.com/facebookresearch/CrypTen) pioneered ML-friendly MPC in PyTorch, but its repository was archived in May 2025 and needs compatibility patches to run on current Python, PyTorch and JAX releases, while recent MPC breakthroughs—[Curl (CAMLIS 2024)](https://eprint.iacr.org/2024/1127), [PUMA (2023)](https://arxiv.org/abs/2307.12533), [SHAFT (NDSS 2025)](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/), and [Fission (IEEE CNS 2025)](https://eprint.iacr.org/2025/653)—live in separate repositories ([jimouris/curl](https://github.com/jimouris/curl), [secretflow/spu](https://github.com/secretflow/spu), [andeskyl/SHAFT](https://github.com/andeskyl/SHAFT)). **PriX** unifies all five in a single, actively maintained JAX/XLA package:

| Feature | CrypTen (PyTorch) | **PriX (JAX/XLA)** |
|---|---|---|
| **Python & Accelerator Support** | Archived (May 2025); needs patches for current Python / PyTorch | **Python 3.10–3.13**, **CPU & CUDA GPU** (tested) via `pip install prix-jax` |
| **API Ergonomics** | `crypten.cryptensor`, `@run_multiprocess`, `crypten.nn` | **`prix.array`, `@prix.mpc.run_multiprocess`, `prix.nn`** (plus drop-in `prix.cryptensor` alias) |
| **Nonlinear Approximation Backends** | Polynomial / NR / limit only (or separate forks) | **5 built-in backends in 1 line** (`mode="lut" \| "poly" \| "shaft" \| "crypten" \| "fission"`) |
| **Kernel Execution** | Eager PyTorch op-by-op launches | **`@jax.jit`-compiled local protocol steps** (fused carry-lookahead rounds, LUT gather, limb matmul, Fourier combination) |
| **Performance (single host, CPU & GPU)** | Baseline (`1.0x`) | Lower latency in every configuration measured. LLM blocks: **2.8x–9.2x faster on CPU**, **2.0x–3.2x faster on GPU**. Primitives: 1.8x–20.8x on CPU, 1.4x–16.2x on GPU ([details](#benchmarks-prix-jax-vs-crypten-pytorch)) |


<p align="center"><img src="docs/figures/llm_latency.png" alt="End-to-end private inference latency of a GPT-2 block, CrypTen vs PriX, on CPU and GPU across five approximation modes" width="95%"/></p>

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
| **`"lut"`** *(default)* | Haar & Biorthogonal-2.2 Wavelet Look-Up Tables + `[EGK+20]` probabilistic truncation (`configs/default.yaml`) | [**Curl (CAMLIS 2024)**](https://eprint.iacr.org/2024/1127) · [jimouris/curl](https://github.com/jimouris/curl) |
| **`"poly"`** | Piecewise low-degree polynomials for `gelu`, `silu`, `exp`, `tanh`, `sigmoid`, `erf` (1 batched comparison + degree-4 polynomial, no LUTs, `configs/poly.yaml`) | [**PUMA (arXiv 2023)**](https://arxiv.org/abs/2307.12533) · [secretflow/spu](https://github.com/secretflow/spu) |
| **`"shaft"`** | 1-round trigonometric-triple Fourier-series GeLU / SiLU / Erf / Tanh / Sigmoid + constant-round Forward-Euler ODE Softmax (`configs/shaft.yaml`) | [**SHAFT (NDSS 2025)**](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/) · [andeskyl/SHAFT](https://github.com/andeskyl/SHAFT) |
| **`"crypten"`** | Classic limit `exp`, Newton–Raphson `reciprocal` / `inv_sqrt`, Taylor `erf`, sigmoid-based `silu`, CrypTen truncation (`configs/approximations.yaml`) | [**CrypTen (NeurIPS 2021)**](https://arxiv.org/abs/2109.00984) · [facebookresearch/CrypTen](https://github.com/facebookresearch/CrypTen) |
| **`"fission"`** | Hybrid permute-and-reveal to a non-colluding plaintext evaluator party, auto-spawned by `prix.run` (`configs/fission.yaml`) | [**Fission (IEEE CNS 2025)**](https://eprint.iacr.org/2025/653) · [jimouris/curl](https://github.com/jimouris/curl) |

### 3. Private Neural Networks & Transformers (`prix.nn`)

Build models with `prix.nn.Module` (`Linear`, `LayerNorm`, `MultiheadAttention`, `GELU`, `SiLU`, `Softmax`, `Sequential`, etc.), call `.encrypt()`, and run private forward passes:

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
| [**02 · Comparing MPC Nonlinearities**](examples/02_approximations_and_ablations.ipynb) | Accuracy & latency ablations across **Curl LUTs**, **PUMA Poly**, **SHAFT**, **CrypTen**, and **Fission**, plus `prix.cfg` hyperparameter sweeps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/02_approximations_and_ablations.ipynb) |
| [**03 · Private Transformers & LLMs (`prix.nn`)**](examples/03_private_transformers_and_llms.ipynb) | Building custom `prix.nn.Module` Transformer blocks, loading `state_dict` weights, and encrypting models with `.encrypt()` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/03_private_transformers_and_llms.ipynb) |
| [**04 · GPU Benchmark Suite (PriX vs. CrypTen)**](examples/prix_vs_curl_gpu_colab.ipynb) | End-to-end reproducible GPU benchmark comparing PriX vs. CrypTen across all 5 modes and 5 LLM architectures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/prix_vs_curl_gpu_colab.ipynb) |

### Guides & Reference Docs
- [**PriX Technical Report / Paper (PDF)**](paper/prix_arxiv.pdf) ([LaTeX source](paper/main.tex)) — *"PriX: Private Accelerated Transformers via Secure Multi-Party Computation in JAX"*.
- [**Quickstart & CrypTen Migration Guide**](docs/quickstart.md) — Side-by-side CrypTen-to-PriX cheat sheet, arithmetic vs. binary secret sharing, and `TFP` vs. `TTP` providers.
- [**MPC Approximation Modes & Configuration Reference**](docs/modes_and_configuration.md) — Deep dive into `"lut"`, `"poly"`, `"shaft"`, `"crypten"`, and `"fission"` modes and how to mix-and-match methods via `prix.cfg`.
- [**Transports & Multi-Host Deployment**](docs/transports.md) — Shared memory vs. peer-to-peer TCP (with optional mutual TLS), running parties on separate machines, and creating certificates.
- [**API Reference**](docs/api_reference.md) — Complete reference for top-level `prix` functions, `prix.Array` methods, and `prix.nn` neural network layers.

---

## Running Tests & Reproducing Benchmarks

To run the unit test suite on CPU:
```bash
JAX_PLATFORMS=cpu python3 -m unittest discover -s tests -q
```

To run the head-to-head benchmark suite against the PyTorch **CrypTen** baseline
([jimouris/curl](https://github.com/jimouris/curl), the CrypTen-based reference implementation of [Curl (CAMLIS 2024)](https://eprint.iacr.org/2024/1127) and [Fission (IEEE CNS 2025)](https://eprint.iacr.org/2025/653), augmented with [PUMA (2023)](https://arxiv.org/abs/2307.12533) and [SHAFT (NDSS 2025)](https://github.com/andeskyl/SHAFT)):
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
> **How to read these numbers.** Both frameworks run all parties as processes on **one host**. There is no network emulation. Both communicate over loopback TCP: CrypTen through `gloo`, PriX through its own peer-to-peer TCP transport (on CPU; the GPU runs predate it and used PriX's shared-memory transport). The [transport comparison](#how-much-does-the-transport-matter) below shows how much the transport matters. Each primitive is the median of 5 timed runs (after 2 warmups), and each LLM block is the median of 3 timed runs (after 1 warmup). Differences of roughly 10% or less are within noise. Benchmarks measure latency only; per-function accuracy is in [Accuracy and Valid Input Ranges](docs/modes_and_configuration.md#5-accuracy-and-valid-input-ranges). On the CrypTen side, the SHAFT functions follow the [SHAFT reference implementation](https://github.com/andeskyl/SHAFT), and the `poly` functions (PUMA has no CrypTen version) are ports of PriX's algorithms. See [`benchmarks/benchmark_prix_vs_crypten.py`](benchmarks/benchmark_prix_vs_crypten.py).

### Charts

<p align="center"><img src="docs/figures/llm_speedup.png" alt="Heatmap of PriX speedup over CrypTen on five Transformer blocks across five approximation modes, CPU and GPU" width="95%"/></p>

<p align="center"><img src="docs/figures/primitive_speedup.png" alt="Heatmap of PriX speedup over CrypTen on ten MPC primitives across five approximation modes, CPU and GPU" width="95%"/></p>

The darkest **Fission** cells partly reflect CPU contention from the extra evaluator process on the CrypTen side (see the notes below). Regenerate the charts from the raw results in [`benchmarks/results/`](benchmarks/results) with `python benchmarks/plot_results.py`.

### How much does the transport matter?

CrypTen and PriX use different transports, so we ran PriX over each of its three transports in the same CPU session as the tables below: its default peer-to-peer TCP transport, CrypTen's own `torch.distributed`/`gloo` transport (by routing PriX's 2-party `all_reduce` through it), and shared memory. Nothing else in PriX changed, and the results are identical.

<p align="center"><img src="docs/figures/transport_ablation.png" alt="Transport comparison: GPT-2 block latency for CrypTen and for PriX over gloo, its own TCP transport and shared memory, plus geometric-mean speedups for each PriX transport" width="95%"/></p>

| PriX speedup over CrypTen (CPU, geomean) | PriX over TCP (default) | PriX over gloo (CrypTen's transport) | PriX over shared memory |
|---|---:|---:|---:|
| Transformer blocks (25 configs) | **4.14x** | 3.14x | 4.31x |
| Primitives (50 configs) | **2.89x** | 1.92x | 3.53x |

- **Most of the speedup does not depend on the transport:** over CrypTen's own `gloo` transport, PriX is still 3.14x faster on Transformer blocks.
- **CrypTen's `gloo` path is itself slow:** PriX's TCP transport is 1.32x faster than `gloo` on blocks and 1.51x on primitives.
- **Shared memory adds little over TCP:** 4% on blocks and 22% on small, round-bound primitives (2% and 8% in a separate interleaved comparison). For openings above 16 MiB, TCP is faster, because shared memory falls back to pickled queues. That's why TCP is the default.
- **Communication is similar but not identical.** In `crypten` mode, rounds agree within one and bytes within 5% (a GPT-2 block takes 176 rounds and sends 80.1 MB with CrypTen versus 79.7 MB with PriX). Two PriX protocol changes explain most other differences: (1) its comparisons open the shared operand of each carry-lookahead AND once instead of twice (a TODO in CrypTen), cutting comparison traffic by about 20% with unchanged security; (2) in `shaft` mode it batches GeLU/SiLU's two comparisons into one, saving 8 rounds per block (98 vs. 106 for GPT-2) versus the SHAFT reference. In `lut` mode PriX uses 2 more rounds per block.

Reproduce with `--comm-stats` and `--prix-transport tcp|gloo|shm` (results under `prix`, `prix_gloo`, `prix_shm`), then `python benchmarks/transport_ablation.py results.json --fig docs/figures/transport_ablation`. Per-workload results, including communication rounds and bytes, are in [`benchmarks/results/`](benchmarks/results).

### 1. CPU Benchmarks

AMD EPYC 7B12 (24 vCPUs), Python 3.13, JAX/jaxlib 0.12.0, PyTorch 2.14.1 (CPU). PriX uses its default peer-to-peer TCP transport and CrypTen uses `gloo` over TCP. Everything was measured in one session.

<details>
<summary><b>CPU tables: LLM latency, speedups, and primitives (click to expand)</b></summary>

#### Private LLM Inference — all modes (ms)
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · Piecewise poly | PriX · Piecewise poly | CrypTen · SHAFT | PriX · SHAFT | CrypTen · Curl LUTs | PriX · Curl LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1423.31 | 582.27 | 1687.14 | 560.17 | 1010.66 | 434.93 | 1106.85 | 446.87 | 443.46 | 163.33 |
| BertBase (1-Block, seq=16) | 2950.74 | 737.85 | 2968.03 | 733.66 | 3154.52 | 735.38 | 2167.78 | 582.69 | 2081.62 | 449.65 |
| GPT2 (1-Block, seq=16) | 3480.96 | 809.60 | 3524.84 | 574.84 | 3915.11 | 519.49 | 2702.19 | 482.07 | 2776.97 | 357.22 |
| Llama (1-Block, d=768, seq=16) | 2317.26 | 566.54 | 2779.46 | 612.08 | 3021.76 | 600.14 | 2068.08 | 525.26 | 2244.43 | 463.35 |
| ModernBert (1-Block, seq=16) | 1899.87 | 557.62 | 2099.36 | 514.43 | 2236.08 | 450.99 | 1685.12 | 425.10 | 1432.81 | 307.39 |

#### PriX speedup over CrypTen — LLM inference
| Model | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **2.44x** | **3.01x** | **2.32x** | **2.48x** | **2.72x** |
| BertBase (1-Block, seq=16) | **4.00x** | **4.05x** | **4.29x** | **3.72x** | **4.63x** |
| GPT2 (1-Block, seq=16) | **4.30x** | **6.13x** | **7.54x** | **5.61x** | **7.77x** |
| Llama (1-Block, d=768, seq=16) | **4.09x** | **4.54x** | **5.04x** | **3.94x** | **4.84x** |
| ModernBert (1-Block, seq=16) | **3.41x** | **4.08x** | **4.96x** | **3.96x** | **4.66x** |

#### MPC Primitives & Nonlinearities on CPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| Private MatMul (128x128) | 23.89 / 5.82 (**4.10x**) | 24.54 / 6.55 (**3.75x**) | 25.05 / 6.15 (**4.08x**) | 28.34 / 7.49 (**3.78x**) | 29.61 / 7.70 (**3.84x**) |
| Private Mul (128x128) | 5.29 / 4.95 (**1.07x**) | 5.89 / 3.65 (**1.61x**) | 4.85 / 4.57 (**1.06x**) | 9.48 / 6.39 (**1.48x**) | 6.90 / 4.47 (**1.54x**) |
| Comparison / ReLU (32x64) | 32.52 / 9.46 (**3.44x**) | 36.94 / 9.50 (**3.89x**) | 31.10 / 10.06 (**3.09x**) | 30.12 / 13.09 (**2.30x**) | 41.98 / 13.25 (**3.17x**) |
| Exp (32x64) | 21.08 / 14.97 (**1.41x**) | 52.61 / 19.06 (**2.76x**) | 20.77 / 17.73 (**1.17x**) | 68.19 / 14.94 (**4.56x**) | 10.90 / 2.67 (**4.08x**) |
| Reciprocal (32x64) | 105.89 / 49.65 (**2.13x**) | 128.86 / 40.38 (**3.19x**) | 95.99 / 59.46 (**1.61x**) | 50.35 / 9.53 (**5.29x**) | 19.56 / 2.67 (**7.33x**) |
| InvSqrt (32x64) | 65.45 / 34.43 (**1.90x**) | 67.89 / 31.04 (**2.19x**) | 69.75 / 45.42 (**1.54x**) | 122.58 / 19.72 (**6.22x**) | 20.26 / 1.53 (**13.25x**) |
| GeLU (32x64) | 114.98 / 62.66 (**1.83x**) | 59.24 / 24.05 (**2.46x**) | 83.85 / 24.27 (**3.45x**) | 34.11 / 8.18 (**4.17x**) | 11.13 / 2.60 (**4.29x**) |
| SiLU (32x64) | 83.53 / 39.35 (**2.12x**) | 53.84 / 23.47 (**2.29x**) | 74.87 / 34.21 (**2.19x**) | 35.45 / 7.40 (**4.79x**) | 12.00 / 1.80 (**6.68x**) |
| Softmax (4x32x32) | 382.48 / 169.38 (**2.26x**) | 442.88 / 159.95 (**2.77x**) | 178.78 / 136.92 (**1.31x**) | 378.89 / 120.87 (**3.13x**) | 18.56 / 2.87 (**6.46x**) |
| LayerNorm (4x32x64) | 75.40 / 28.37 (**2.66x**) | 72.25 / 31.84 (**2.27x**) | 76.07 / 40.36 (**1.88x**) | 55.58 / 13.91 (**4.00x**) | 21.72 / 3.07 (**7.07x**) |

</details>

> [!NOTE]
> The host is a shared cloud VM. Absolute latencies vary between sessions (both frameworks ran about 25–40% slower in this session than in an earlier one), so compare numbers only within one session. CrypTen's ReLU, whose protocol is identical in every mode, took 30–37 ms in four modes and 42 ms in `fission` mode, where the extra evaluator process competes for the host.

---

### 2. GPU Benchmarks (NVIDIA Tesla T4, CUDA)

Google Colab, one Tesla T4. All five modes were measured in one session.

<details>
<summary><b>GPU tables: LLM latency, speedups, and primitives (click to expand)</b></summary>

#### Private LLM Inference — all modes (ms)
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · Piecewise poly | PriX · Piecewise poly | CrypTen · SHAFT | PriX · SHAFT | CrypTen · Curl LUTs | PriX · Curl LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 2059.81 | 908.87 | 2002.88 | 773.85 | 1306.09 | 492.68 | 2102.56 | 660.26 | 415.48 | 155.49 |
| BertBase (1-Block, seq=16) | 1729.39 | 773.28 | 1682.48 | 765.60 | 1372.00 | 657.27 | 1776.27 | 670.65 | 932.96 | 423.73 |
| GPT2 (1-Block, seq=16) | 1715.69 | 802.80 | 1662.84 | 760.49 | 1335.14 | 629.89 | 1690.98 | 670.36 | 863.79 | 437.04 |
| Llama (1-Block, d=768, seq=16) | 1684.07 | 638.33 | 1673.81 | 640.89 | 1316.48 | 501.02 | 1735.99 | 559.14 | 1016.31 | 350.65 |
| ModernBert (1-Block, seq=16) | 1538.26 | 572.18 | 1433.70 | 547.52 | 1121.48 | 407.00 | 1503.04 | 484.17 | 704.84 | 240.73 |

#### PriX speedup over CrypTen — LLM inference on GPU
| Model | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **2.27x** | **2.59x** | **2.65x** | **3.18x** | **2.67x** |
| BertBase (1-Block, seq=16) | **2.24x** | **2.20x** | **2.09x** | **2.65x** | **2.20x** |
| GPT2 (1-Block, seq=16) | **2.14x** | **2.19x** | **2.12x** | **2.52x** | **1.98x** |
| Llama (1-Block, d=768, seq=16) | **2.64x** | **2.61x** | **2.63x** | **3.10x** | **2.90x** |
| ModernBert (1-Block, seq=16) | **2.69x** | **2.62x** | **2.76x** | **3.10x** | **2.93x** |

#### MPC Primitives & Nonlinearities on GPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| Private MatMul (128x128) | 14.21 / 4.94 (**2.88x**) | 13.32 / 4.20 (**3.17x**) | 11.26 / 4.15 (**2.71x**) | 17.66 / 12.22 (**1.45x**) | 16.60 / 3.57 (**4.65x**) |
| Private Mul (128x128) | 4.60 / 2.55 (**1.80x**) | 5.40 / 2.49 (**2.17x**) | 4.69 / 2.76 (**1.70x**) | 9.77 / 5.62 (**1.74x**) | 8.56 / 2.68 (**3.19x**) |
| Comparison / ReLU (32x64) | 50.83 / 14.04 (**3.62x**) | 54.01 / 13.82 (**3.91x**) | 45.08 / 13.98 (**3.22x**) | 44.58 / 13.73 (**3.25x**) | 73.41 / 13.67 (**5.37x**) |
| Exp (32x64) | 32.79 / 18.03 (**1.82x**) | 84.77 / 29.61 (**2.86x**) | 33.40 / 18.02 (**1.85x**) | 70.83 / 25.49 (**2.78x**) | 7.41 / 1.73 (**4.28x**) |
| Reciprocal (32x64) | 141.59 / 64.81 (**2.18x**) | 141.50 / 65.19 (**2.17x**) | 147.21 / 62.27 (**2.36x**) | 46.67 / 11.96 (**3.90x**) | 18.67 / 1.81 (**10.31x**) |
| InvSqrt (32x64) | 89.88 / 43.79 (**2.05x**) | 96.30 / 39.39 (**2.44x**) | 91.43 / 41.59 (**2.20x**) | 195.46 / 12.08 (**16.18x**) | 19.51 / 1.75 (**11.15x**) |
| GeLU (32x64) | 152.44 / 76.06 (**2.00x**) | 70.33 / 32.44 (**2.17x**) | 96.21 / 25.45 (**3.78x**) | 26.97 / 12.88 (**2.09x**) | 7.83 / 1.88 (**4.16x**) |
| SiLU (32x64) | 148.35 / 57.28 (**2.59x**) | 71.20 / 32.45 (**2.19x**) | 94.54 / 29.58 (**3.20x**) | 27.53 / 12.71 (**2.17x**) | 7.10 / 1.62 (**4.38x**) |
| Softmax (4x32x32) | 586.17 / 256.97 (**2.28x**) | 593.94 / 226.88 (**2.62x**) | 215.84 / 93.65 (**2.30x**) | 564.97 / 191.22 (**2.95x**) | 8.29 / 1.48 (**5.60x**) |
| LayerNorm (4x32x64) | 104.48 / 46.18 (**2.26x**) | 107.81 / 42.79 (**2.52x**) | 104.40 / 47.23 (**2.21x**) | 184.10 / 24.14 (**7.63x**) | 8.85 / 2.17 (**4.08x**) |

</details>

> [!NOTE]
> As on CPU, CrypTen's `fission`-mode ReLU (73 ms) is slower than in the other modes (45–54 ms), even though the protocol is identical. PriX's MatMul is slower in `lut` mode (12.2 ms vs. 3.6–4.9 ms), where products use probabilistic truncation.



---

## References

PriX implements and accelerates the protocols and nonlinear approximations introduced in the following works:

- **CrypTen (`mode="crypten"`)** — Brian Knott, Shobha Venkataraman, Awni Hannun, Shubho Sengupta, Mark Ibrahim, and Laurens van der Maaten. *"CrypTen: Secure Multi-Party Computation Meets Machine Learning."* **NeurIPS 2021** ([Paper](https://arxiv.org/abs/2109.00984) · [Code](https://github.com/facebookresearch/CrypTen)).
- **Curl (`mode="lut"`)** — Manuel B. Santos, Dimitris Mouris, Mehmet Ugurbil, Stanislaw Jarecki, José Reis, Shubho Sengupta, and Miguel de Vega. *"Curl: Private LLMs through Wavelet-Encoded Look-Up Tables."* **CAMLIS 2024** ([ePrint 2024/1127](https://eprint.iacr.org/2024/1127) · [Code](https://github.com/jimouris/curl)).
- **PUMA (`mode="poly"`)** — Ye Dong, Wen-jie Lu, Yancheng Zheng, Haoqi Wu, Derun Zhao, Jin Tan, Zhicong Huang, Cheng Hong, Tao Wei, and Wenguang Chen. *"PUMA: Secure Inference of LLaMA-7B in Five Minutes."* **arXiv 2023** ([arXiv:2307.12533](https://arxiv.org/abs/2307.12533) · [Code](https://github.com/secretflow/spu)).
- **SHAFT (`mode="shaft"`)** — Andes Y. L. Kei and Sherman S. M. Chow. *"SHAFT: Secure, Handy, Accurate, and Fast Transformer Inference."* **NDSS 2025** ([Paper](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/) · [ePrint 2025/2324](https://eprint.iacr.org/2025/2324) · [Code](https://github.com/andeskyl/SHAFT)).
- **Fission (`mode="fission"`)** — Mehmet Ugurbil, Dimitris Mouris, Manuel B. Santos, José Cabrero-Holgueras, Miguel de Vega, and Shubho Sengupta. *"Fission: Distributed Privacy-Preserving Large Language Model Inference."* **IEEE CNS 2025** ([ePrint 2025/653](https://eprint.iacr.org/2025/653) · [Code](https://github.com/jimouris/curl)).

### Citing PriX

If you use **PriX** in your research, please cite our technical report ([`paper/prix_arxiv.pdf`](paper/prix_arxiv.pdf)):

```bibtex
@misc{ugurbil2026prix,
  author       = {Mehmet Ugurbil},
  title        = {{PriX}: Private Accelerated Transformers via Secure Multi-Party Computation in {JAX}},
  howpublished = {Cryptology ePrint Archive},
  year         = {2026},
  url          = {https://github.com/mugurbil/prix}
}
```

---

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [JAX](https://github.com/jax-ml/jax).
