# PriX Benchmarks

All numbers below are reproducible from [`benchmarks/`](../benchmarks). Sections 1–2 compare PriX with the PyTorch CrypTen reference; Section 3 compares the FSS / wavelet modes (default, Wave, SIGMA, Curl, SHAFT) on rounds, online and preprocessing bytes, latency and accuracy.

## PriX (JAX) vs. CrypTen (PyTorch)

> [!NOTE]
> **How to read these numbers.** Both frameworks run all parties as processes on **one host**. There is no network emulation. Both communicate over loopback TCP: CrypTen through `gloo`, PriX through its own peer-to-peer TCP transport. The [transport comparison](#how-much-does-the-transport-matter) below shows how much the transport matters. Each primitive is the median of 5 timed runs (after 2 warmups), and each LLM block is the median of 3 timed runs (after 1 warmup). Differences of roughly 10% or less are within noise. Benchmarks measure latency only; per-function accuracy is in [Accuracy and Valid Input Ranges](modes_and_configuration.md#5-accuracy-and-valid-input-ranges). On the CrypTen side, the SHAFT functions follow the [SHAFT reference implementation](https://github.com/andeskyl/SHAFT), and the `poly` functions (PUMA has no CrypTen version) are ports of PriX's algorithms. See [`benchmarks/benchmark_prix_vs_crypten.py`](../benchmarks/benchmark_prix_vs_crypten.py).

### Charts

<p align="center"><img src="figures/llm_speedup.png" alt="Heatmap of PriX speedup over CrypTen on five Transformer blocks across five approximation modes, CPU and GPU" width="95%"/></p>

<p align="center"><img src="figures/primitive_speedup.png" alt="Heatmap of PriX speedup over CrypTen on ten MPC primitives across five approximation modes, CPU and GPU" width="95%"/></p>

The darkest **Fission** cells partly reflect CPU contention from the extra evaluator process on the CrypTen side (see the notes below). Regenerate the charts from the raw results in [`benchmarks/results/`](../benchmarks/results) with `python benchmarks/plot_results.py`.

### How much does the transport matter?

CrypTen and PriX use different transports, so we ran PriX over each of its three transports in the same CPU and GPU sessions as the tables below: its default peer-to-peer TCP transport, CrypTen's own `torch.distributed`/`gloo` transport (by routing PriX's 2-party `all_reduce` through it), and shared memory. Nothing else in PriX changed, and the results are identical.

<p align="center"><img src="figures/transport_ablation.png" alt="Transport comparison: GPT-2 block latency for CrypTen and for PriX over gloo, its own TCP transport and shared memory, plus geometric-mean speedups for each PriX transport" width="95%"/></p>

| PriX speedup over CrypTen (geomean) | PriX over TCP (default) | PriX over gloo (CrypTen's transport) | PriX over shared memory |
|---|---:|---:|---:|
| CPU · Transformer blocks (25 configs) | **4.14x** | 3.14x | 4.31x |
| CPU · Primitives (50 configs) | **2.89x** | 1.92x | 3.53x |
| GPU · Transformer blocks (25 configs) | **2.33x** | 2.17x | 2.46x |
| GPU · Primitives (50 configs) | **2.69x** | 2.17x | 2.85x |

- **Most of the speedup does not depend on the transport:** over CrypTen's own `gloo` transport, PriX is still 3.14x faster on Transformer blocks on CPU and 2.17x on GPU.
- **CrypTen's `gloo` path is itself slow:** PriX's TCP transport is 1.32x faster than `gloo` on CPU blocks (1.51x on primitives) and 1.07x faster on GPU blocks (1.24x on primitives).
- **Shared memory adds little over TCP:** 4–5% on blocks and 22% (CPU) / 6% (GPU) on small, round-bound primitives. For openings above 16 MiB, TCP is faster, because shared memory falls back to pickled queues. That's why TCP is the default.
- **Communication is similar but not identical.** In `crypten` mode, rounds agree within one and bytes within 5% (a GPT-2 block takes 176 rounds and sends 80.1 MB with CrypTen versus 79.7 MB with PriX). Two PriX protocol changes explain most other differences: (1) its comparisons open the shared operand of each carry-lookahead AND once instead of twice (a TODO in CrypTen), cutting comparison traffic by about 20% with unchanged security; (2) in `shaft` mode it batches GeLU/SiLU's two comparisons into one, saving 8 rounds per block (98 vs. 106 for GPT-2) versus the SHAFT reference. In `curl` mode PriX uses 2 more rounds per block.

Reproduce with `--comm-stats` and `--prix-transport tcp|gloo|shm` (results under `prix`, `prix_gloo`, `prix_shm`), then `python benchmarks/transport_ablation.py results.json --fig docs/figures/transport_ablation`. Per-workload results, including communication rounds and bytes, are in [`benchmarks/results/`](../benchmarks/results).

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

Google Colab, one Tesla T4. PriX uses its default peer-to-peer TCP transport and CrypTen uses `gloo` over TCP. All five modes were measured in one session.

<details>
<summary><b>GPU tables: LLM latency, speedups, and primitives (click to expand)</b></summary>

#### Private LLM Inference — all modes (ms)
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · Piecewise poly | PriX · Piecewise poly | CrypTen · SHAFT | PriX · SHAFT | CrypTen · Curl LUTs | PriX · Curl LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1935.00 | 867.19 | 1815.61 | 826.13 | 1197.88 | 537.35 | 1968.48 | 639.19 | 396.46 | 155.38 |
| BertBase (1-Block, seq=16) | 1662.57 | 761.04 | 1676.78 | 763.90 | 1321.44 | 650.66 | 1694.65 | 600.76 | 926.40 | 350.81 |
| GPT2 (1-Block, seq=16) | 1663.77 | 712.28 | 1609.94 | 794.24 | 1290.56 | 645.33 | 1634.65 | 604.11 | 838.81 | 320.92 |
| Llama (1-Block, d=768, seq=16) | 1632.13 | 739.13 | 1580.11 | 729.44 | 1286.35 | 603.96 | 1728.45 | 658.25 | 983.09 | 402.74 |
| ModernBert (1-Block, seq=16) | 1445.26 | 686.65 | 1390.71 | 628.84 | 1085.58 | 568.86 | 1480.95 | 531.07 | 689.95 | 297.91 |

#### PriX speedup over CrypTen — LLM inference on GPU
| Model | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **2.23x** | **2.20x** | **2.23x** | **3.08x** | **2.55x** |
| BertBase (1-Block, seq=16) | **2.18x** | **2.20x** | **2.03x** | **2.82x** | **2.64x** |
| GPT2 (1-Block, seq=16) | **2.34x** | **2.03x** | **2.00x** | **2.71x** | **2.61x** |
| Llama (1-Block, d=768, seq=16) | **2.21x** | **2.17x** | **2.13x** | **2.63x** | **2.44x** |
| ModernBert (1-Block, seq=16) | **2.10x** | **2.21x** | **1.91x** | **2.79x** | **2.32x** |

#### MPC Primitives & Nonlinearities on GPU (ms, CrypTen / PriX)
| Benchmark | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| Private MatMul (128x128) | 11.45 / 4.35 (**2.63x**) | 12.66 / 3.07 (**4.12x**) | 11.87 / 3.61 (**3.29x**) | 16.47 / 7.71 (**2.14x**) | 11.13 / 3.64 (**3.05x**) |
| Private Mul (128x128) | 4.91 / 3.23 (**1.52x**) | 4.51 / 3.12 (**1.45x**) | 4.08 / 3.31 (**1.23x**) | 8.59 / 5.42 (**1.59x**) | 4.99 / 2.89 (**1.73x**) |
| Comparison / ReLU (32x64) | 50.43 / 15.73 (**3.21x**) | 44.18 / 15.18 (**2.91x**) | 41.51 / 15.77 (**2.63x**) | 40.80 / 14.97 (**2.73x**) | 45.86 / 15.65 (**2.93x**) |
| Exp (32x64) | 28.30 / 19.07 (**1.48x**) | 79.72 / 31.14 (**2.56x**) | 27.03 / 17.71 (**1.53x**) | 67.40 / 28.30 (**2.38x**) | 6.85 / 1.97 (**3.49x**) |
| Reciprocal (32x64) | 129.92 / 65.24 (**1.99x**) | 134.66 / 68.01 (**1.98x**) | 137.68 / 69.79 (**1.97x**) | 56.21 / 12.59 (**4.47x**) | 17.61 / 1.48 (**11.87x**) |
| InvSqrt (32x64) | 90.38 / 39.73 (**2.28x**) | 87.97 / 38.64 (**2.28x**) | 85.92 / 40.36 (**2.13x**) | 141.19 / 12.20 (**11.57x**) | 17.45 / 1.52 (**11.49x**) |
| GeLU (32x64) | 150.42 / 73.75 (**2.04x**) | 68.45 / 37.10 (**1.84x**) | 92.71 / 29.91 (**3.10x**) | 23.39 / 12.35 (**1.89x**) | 6.81 / 1.71 (**3.98x**) |
| SiLU (32x64) | 132.88 / 50.49 (**2.63x**) | 67.45 / 36.06 (**1.87x**) | 90.73 / 27.87 (**3.26x**) | 26.20 / 12.55 (**2.09x**) | 7.30 / 1.71 (**4.26x**) |
| Softmax (4x32x32) | 551.32 / 251.27 (**2.19x**) | 586.34 / 252.38 (**2.32x**) | 199.67 / 106.15 (**1.88x**) | 507.15 / 200.35 (**2.53x**) | 7.23 / 1.74 (**4.15x**) |
| LayerNorm (4x32x64) | 103.48 / 43.67 (**2.37x**) | 103.03 / 47.66 (**2.16x**) | 99.32 / 49.61 (**2.00x**) | 158.90 / 24.92 (**6.38x**) | 9.46 / 2.48 (**3.81x**) |

</details>

> [!NOTE]
> CrypTen's ReLU took 41–50 ms across the five modes; PriX's took 15–16 ms. PriX's MatMul is slower in `curl` mode (7.7 ms vs. 3.1–4.4 ms), where products use probabilistic truncation.

---

### 3. Function Secret Sharing & Wavelets: Wave vs. SIGMA, Curl, and SHAFT

In addition to the five CrypTen-compatible modes above, PriX implements 2-party **Function Secret Sharing (FSS)** in [`prix/fss.py`](prix/fss.py) for [**SIGMA (`mode="sigma"`)**](https://doi.org/10.56553/popets-2024-0107) and [**Wave (`mode="wave"`)**](https://doi.org/10.56553/popets-2025-0083), supporting both deterministic DPF/DCF LUT truncation (`lut_indexing="wave"`) and 0-round probabilistic share splitting (`lut_indexing="wave_prob"`). See **Section 6 of [`paper/prix_arxiv.pdf`](../paper/prix_arxiv.pdf)** for the full study.

| Model / Sublayer | Metric | **PriX default** (`mode="default"`) | **Wave** (`mode="wave"`) | **Wave-Prob** (`lut_indexing="wave_prob"`) | **SIGMA** (`mode="sigma"`) | **Curl** (`mode="curl"`) | **SHAFT** (`mode="shaft"`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`BERT-base` GeLU** `(1, 128, 3072)` | Rounds / 2P Comm | 3 rds / 37.7 MB | **2 rds** / **31.5 MB** | **2 rds** / **31.5 MB** | 6 rds / 44.2 MB | 4 rds / **31.5 MB** | 11 rds / 427.8 MB |
| | Local CPU Time / MAE | 0.67 s / 0.00045 | 1.66 s / 0.00190 | 0.21 s / 0.00984 | 2.60 s / **0.00038** | **0.13 s** / 0.00982 | 0.52 s / 0.00098 |
| **`BERT-base` Softmax** `(1, 12, 128, 128)` | Rounds / 2P Comm | 25 rds / 38.0 MB | **23 rds** / **34.8 MB** | **23 rds** / **34.8 MB** | 31 rds / 56.8 MB | 98 rds / 311.9 MB | 41 rds / 346.0 MB |
| | Local CPU Time / MAE | 0.77 s / 6.9e-5 | 1.77 s / 6.9e-5 | 0.74 s / 2.8e-4 | 2.16 s / **1.1e-5** | 0.64 s / 2.8e-4 | **0.63 s** / 3.0e-4 |
| **`BERT-base` (12L, $L=128$)** | Rounds (Total) | 828 | **696** | **696** | 1,032 | 1,668 | 1,200 |
| | Online Comm 1P / Key 1P ($\lambda$=128) | 0.83 GB / 8.1 GB | **0.78 GB** / 10.9 GB | **0.78 GB** / **6.6 GB** | 0.98 GB / 13.9 GB | 3.12 GB / 6.4 GB | 5.51 GB / 9.1 GB |
| | Local CPU Time / Block MAE | 24.2 s / **0.0007** | 50.6 s / 0.0028 | **17.7 s** / 0.0191 | 65.9 s / 0.0017 | 18.0 s / 0.0191 | 20.5 s / 0.0015 |
| **`BERT-large` (24L, $L=128$)** | Rounds (Total) | 1,656 | **1,392** | **1,392** | 2,064 | 3,336 | 2,400 |
| | Online Comm 1P / Key 1P ($\lambda$=128) | 2.22 GB / 21.5 GB | **2.07 GB** / 29.2 GB | **2.07 GB** / **17.5 GB** | 2.62 GB / 37.0 GB | 8.92 GB / 17.7 GB | 15.30 GB / 24.9 GB |
| | Local CPU Time / Block MAE | 65.3 s / **0.0007** | 140.0 s / 0.0028 | **49.4 s** / 0.0192 | 189.4 s / 0.0016 | 55.3 s / 0.0192 | 61.8 s / 0.0015 |
| **`GPT-2` (12L, $L=64$)** | Rounds (Total) | 804 | **672** | **672** | 1,008 | 1,656 | 1,200 |
| | Online Comm 1P / Key 1P ($\lambda$=128) | 0.35 GB / 2.8 GB | **0.32 GB** / 3.8 GB | **0.32 GB** / **1.9 GB** | 0.39 GB / 5.3 GB | 1.39 GB / 2.6 GB | 2.57 GB / 4.1 GB |
| | Local CPU Time / Block MAE | 12.2 s / **0.0007** | 24.2 s / 0.0028 | **8.5 s** / 0.0193 | 27.7 s / 0.0015 | 12.9 s / 0.0194 | 12.3 s / 0.0015 |

> [!IMPORTANT]
> **Input range matters more than table resolution.** Before v0.2.1 every LUT mode (Curl, SIGMA, Wave) had a Transformer-block MAE of 0.045–0.051 regardless of how accurate its primitives were, because the softmax denominator $\sum_j e^{x_j-\max} \in [1, L]$ exceeded the reciprocal table's $[0, 64)$ domain whenever attention was near-uniform (rows summed to 9,903 instead of 1 at logit scale 0.3), and LayerNorm's `inv_sqrt` table stopped at variance 16. PriX now sizes the inverse table from the softmax length (as SIGMA §5.2.3 does) and evaluates `inv_sqrt` with a range-extended scheme (`method: "bior_scaled"`: one bior(5,3) table applied to $x$, $2^8x$, $2^{16}x$ in a single batched LUT call, window selected by two batched FSS comparisons) that is accurate for variances in $[2^{-8}, 2^{16})$ at +3 rounds per LayerNorm; block MAE dropped $60\times$. See Section 6.3 of the paper.

- **Wave vs. SIGMA (`mode="sigma"`):** `mode="sigma"` implements SIGMA's exact two-phase protocols (pre-LUT DPF sign/clip comparisons + $\Pi^{\text{Select}}$ mux + 1-round $\Pi^{\text{TR}}_{n,f}$ + 2-round DPF-LUT on $2^8\text{–}2^{10}$ step tables, two-table $2^8\times 2^8$ `nexp`, and tailored `inv_sqrt`, matching SIGMA Table 2, Table 3 & Table 5 communication: `56.3 B`/elem vs. `58 B` on GeLU/SiLU, `0.265 GB` vs. `0.26 GB` 1P on 12L `BERT-base` GeLU, `0.341 GB` vs. `0.37 GB` 1P on Softmax, and `0.984 GB` vs. `0.99 GB` 1P on `BERT-base`). Because **Wave** encodes the full activation domain directly into compactly supported 1st-order $\text{bior}(5,3)$ wavelet tables ($16\text{–}32$ intervals) and folds exact bin truncation into the $(J+\mathfrak{j})$-level DPF + $\mathfrak{j}$-level arithmetic DCF, Wave eliminates SIGMA's pre-LUT comparisons, muxes, and $\Pi^{\text{TR}}$ round—cutting Transformer rounds by **$1.48\times$** (`58` vs. `86` rounds/block; `2` vs. `6` on GeLU/SiLU; `23` vs. `31` on Softmax), online communication by **$21\%\text{–}39\%$** (`1.551 GB` vs. `1.968 GB` on `BERT-base`; `34.82 MB` vs. `56.82 MB` on Softmax), and local CPU runtime by **$1.3\times\text{–}3.5\times$** (`50.6s` vs. `65.9s` on `BERT-base`; `6.9ms` vs. `24.3ms` on SiLU) — **$3\times$** with the default preset's SIGMA-TR indexing (`21.7s`).
- **Wave vs. Curl (`mode="curl"`):** Wave cuts Transformer rounds by **$2.4\times$** (`58` vs. `139` rounds/block) and total communication by **$4.0\times$** (`1.55 GB` vs. `6.23 GB` on `BERT-base`), while eliminating off-by-1 probabilistic knot boundary jumps.
- **Wave & SIGMA vs. SHAFT (`mode="shaft"`):** SHAFT's Fourier series requires 11 comparison rounds on unbounded activations (`GeLU`/`SiLU`), sending **$5.6\times\text{–}7.1\times$ more traffic** (`11.02 GB` vs. `1.97 GB` for SIGMA and `1.55 GB` for Wave on `BERT-base`) and **$1.16\times\text{–}1.72\times$ more rounds** (`1,200` vs. `1,032` and `696`).
- **Deterministic vs. Probabilistic Truncation (`lut_indexing="wave"` vs. `"wave_prob"`, `truncation="egk"` vs. `"crypten"`):** Switching `encoder.lut_indexing` to `"wave_prob"` removes the $\mathfrak{j}$-level DPF tail and arithmetic DCF, accelerating local GeLU by **$7.8\times$** and `BERT-base` by **$2.9\times$** (`17.7s`, faster than both SHAFT and Curl even in local CPU compute), at the cost of $5\times\text{–}13\times$ higher LUT boundary error (block MAE `0.019`, matching Curl). Conversely, 0-round signed `int64` product shift (`truncation="crypten"`) has a 1-ULP LSB carry and MSB wrap probability $\frac{|Z|}{2^{64}} = \frac{|x \cdot y|\cdot 2^{2f}}{2^{64}}$ ($\sim 10^{-10}$ on small single-block tensors, but prone to catastrophic $\pm 2^{32}$ spikes on large activations, $N \ge 3$ parties, or after modular-wrapping protocols; see Section 6.4 of the paper).

#### Preprocessing key size & Wave with SIGMA's Truncate-Reduce (`mode="wave"`, `lut_indexing="sigma"`)

> [!WARNING]
> All latencies above are **online-phase** numbers with a **simulated dealer** (`provider=TFP`): both parties expand the same PRG seeds, so FSS correction words, Beaver tuples and masks are derived locally and **no key material is transmitted**. This matches the convention of CrypTen's `TrustedFirstParty` and of the online columns in the SIGMA/Wave papers, but it is **not a secure deployment** and it hides the dealer's key-transfer cost. SIGMA's published *GPU* latencies include host→device key transfer, so PriX's device-local timings must not be compared against them—compare rounds, online bytes, key bytes and accuracy instead.

`prix.provider.get_preproc_stats()` reports the key bytes one party would receive from a real dealer (seed correction words re-counted at $\lambda=128$, i.e. 16 B / DPF level, for comparability with SIGMA/Orca). The accounting reproduces SIGMA Table 2 (`1,488 B` vs. `1.43 KB` on GeLU, `1,888 B` vs. `1.93 KB` on Rsqrt). Run [`benchmarks/benchmark_fss_modes.py`](../benchmarks/benchmark_fss_modes.py) (`--device cuda` for GPU; Colab: [`examples/prix_fss_wave_sigma_gpu_colab.ipynb`](../examples/prix_fss_wave_sigma_gpu_colab.ipynb)) to reproduce:

| `BERT-base` (12L, $L=128$), per party | **PriX default** | **Wave** (`lut_indexing="wave"`) | **Wave + SIGMA TR** (`lut_indexing="sigma"`) | **Wave-Prob** (`lut_indexing="wave_prob"`) | **SIGMA** | **Curl** | **SHAFT** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online rounds | 828 | **696** | 756 | **696** | 1,032 | 1,668 | 1,200 |
| Online bytes (1P) | 0.832 GB | **0.775 GB** | 0.832 GB | **0.775 GB** | 0.984 GB | 3.117 GB | 5.512 GB |
| Preprocessing key (1P, $\lambda=128$) | 8.06 GB | 10.94 GB | 9.30 GB | **6.57 GB** | 13.89 GB | 6.43 GB | 9.12 GB |
| Local CPU time (12L) | 24.2 s | 50.6 s | 22.6 s | **17.7 s** | 65.9 s | 18.0 s | 20.5 s |
| Block MAE | **0.0007** | 0.0028 | 0.0028 | 0.0191 | 0.0017 | 0.0191 | 0.0015 |

- **Wave + SIGMA TR** replaces the $(J+\mathfrak{j})$-level Grotto carry loop (evaluated at all $2^J$ leaves) with SIGMA's single-path $\mathfrak{j}$-level DCF followed by a $J$-level DPF, keeping Wave's exact deterministic accuracy while cutting PriX's CPU time by **$2.2\times$** on `BERT-base` (`50.6 s` → `22.6 s`), **$3.4\times$** on GeLU `(1,128,3072)` (`1.66 s` → `0.49 s`) and **$4.6\times$** on `inv_sqrt`, and the key by $15\%$–$30\%$—for `+1` round and `+8 B`/elem per LUT. Versus `mode="sigma"` it needs $1.36\times$ fewer rounds, $15\%$ less online traffic, $33\%$ less key and $2.9\times$ less CPU time. The **default preset** adds 32-knot GeLU/SiLU tables (4× lower activation error for +32 B key, no extra round) and 48-bit comparisons (SIGMA's effective bit-width, −24% PRG calls), reaching block MAE **0.0007** — the most accurate mode in PriX — at `24.2 s` (828 rounds, including the range-extended LayerNorm).
- **Preprocessing dominates every FSS scheme** (key ≈ $8.5\times$–$14\times$ online bytes), so total bytes is the honest metric: Wave-Prob `7.3 GB` < default `8.9 GB` < Curl `9.5 GB` < Wave + SIGMA TR `10.1 GB` < Wave `11.7 GB` < SHAFT `14.6 GB` < SIGMA `14.9 GB`.

---
