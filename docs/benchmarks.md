# PriX Benchmarks

Every table on this page is generated from `benchmarks/results/*.json` by `python benchmarks/make_tables.py`; the same files feed the paper. [`docs/provenance.md`](provenance.md) lists which script and commit produced each one. To refresh everything on a CPU host: `CRYPTEN_ROOT=/path/to/jimouris/curl bash benchmarks/run_all.sh`; GPU numbers come from the two Colab notebooks ([04](../examples/prix_vs_curl_gpu_colab.ipynb), [05](../examples/prix_fss_wave_sigma_gpu_colab.ipynb)).

> [!NOTE]
> **How to read these numbers.** All parties run as processes on **one host**; there is no network emulation, so latencies measure computation plus loopback communication. Each primitive is the median of 5 timed runs after 2 warm-ups, each Transformer block the median of 5 runs after 1 warm-up. The CPU host is a shared cloud VM with an ambient load of 8–12; absolute latencies vary by 10–30 % between sessions, so compare numbers within one session (one results file) and treat differences below ~15 % as noise. Rounds, bytes, key sizes and errors are deterministic.

## 1. PriX (JAX) vs. CrypTen (PyTorch)

The PyTorch baseline is [jimouris/curl](https://github.com/jimouris/curl) (the CrypTen fork that is the reference implementation of Curl and Fission) augmented with the [SHAFT](https://github.com/andeskyl/SHAFT) reference functions; the `poly` functions (PUMA has no CrypTen version) are ports of PriX's algorithms. Both frameworks run the same five modes with identical configuration; CrypTen communicates through `gloo`, PriX through its own peer-to-peer TCP transport (see §1.1).

<p align="center"><img src="figures/llm_latency.png" alt="GPT-2 block latency, CrypTen vs PriX, CPU and GPU, five modes" width="95%"/></p>
<p align="center"><img src="figures/llm_speedup.png" alt="PriX speedup over CrypTen on five Transformer blocks" width="95%"/></p>
<p align="center"><img src="figures/primitive_speedup.png" alt="PriX speedup over CrypTen on ten primitives" width="95%"/></p>

### 1.1 How much does the transport matter?

PriX was also run over CrypTen's own `torch.distributed`/`gloo` transport (its 2-party `all_reduce` routed through gloo) and over shared memory, in the same sessions. Nothing else changes.

<p align="center"><img src="figures/transport_ablation.png" alt="Transport ablation" width="95%"/></p>

<!-- generated:transport:begin -->
| PriX speedup over CrypTen (geomean) | PriX over TCP (default) | PriX over gloo (CrypTen's transport) | PriX over shared memory |
|---|---:|---:|---:|
| CPU · Transformer blocks (25 configs) | **4.02x** | 3.39x | 4.42x |
| CPU · Primitives (50 configs) | **2.81x** | 2.16x | 3.53x |
| GPU · Transformer blocks (25 configs) | **2.26x** | 2.23x | 2.46x |
| GPU · Primitives (50 configs) | **2.66x** | 2.34x | 2.98x |

- **CPU:** over CrypTen's own `gloo` transport PriX is still 3.39x faster on Transformer blocks (2.16x on primitives); PriX's TCP transport is 1.19x faster than `gloo` on blocks (1.30x on primitives); shared memory adds 10% on blocks and 26% on small, round-bound primitives.
- **GPU:** over CrypTen's own `gloo` transport PriX is still 2.23x faster on Transformer blocks (2.34x on primitives); PriX's TCP transport is 1.01x faster than `gloo` on blocks (1.14x on primitives); shared memory adds 9% on blocks and 12% on small, round-bound primitives.
<!-- generated:transport:end -->

- **Communication is similar but not identical.** In `crypten` mode, rounds agree within one and bytes within 5 %. Two PriX protocol changes explain most other differences: its comparisons open the shared operand of each carry-lookahead AND once instead of twice (a TODO in CrypTen), and in `shaft` mode it batches GeLU/SiLU's two comparisons into one. Shared memory falls back to pickled queues above 16 MiB per opening, which is why TCP is the default.

### 1.2 CPU

<!-- generated:cpu_env:begin -->
cpu_model: `AMD EPYC 7B12` · python_version: `3.13.15` · jax_version: `0.12.0` · torch_version: `2.14.1+cpu` · prix_commit: `56a6672c0151c0f9decab07b8951d45a63052bb8` · timestamp_utc: `2026-10-24T02:49:33Z` · loadavg_1m_at_start: `9.27`
<!-- generated:cpu_env:end -->

#### Transformer blocks: latency (ms)
<!-- generated:cpu_llm_latency:begin -->
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · Piecewise poly | PriX · Piecewise poly | CrypTen · SHAFT (Fourier + ODE) | PriX · SHAFT (Fourier + ODE) | CrypTen · Curl LUTs | PriX · Curl LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 1321.25 | 856.40 | 1603.67 | 492.42 | 1424.65 | 390.16 | 969.74 | 416.80 | 311.25 | 136.21 |
| BertBase (1-Block, seq=16) | 2882.67 | 1122.40 | 2873.30 | 646.21 | 2985.11 | 545.46 | 2233.01 | 548.13 | 1627.21 | 345.32 |
| GPT2 (1-Block, seq=16) | 3601.80 | 883.03 | 3432.13 | 606.49 | 3389.43 | 510.91 | 2753.30 | 494.45 | 2052.97 | 335.22 |
| Llama (1-Block, d=768, seq=16) | 2270.18 | 639.89 | 2574.41 | 517.22 | 2662.41 | 495.31 | 2097.84 | 485.81 | 1628.23 | 341.82 |
| ModernBert (1-Block, seq=16) | 1730.70 | 523.73 | 2078.42 | 502.62 | 1990.67 | 408.75 | 1552.91 | 400.21 | 1147.84 | 272.87 |
<!-- generated:cpu_llm_latency:end -->

#### Transformer blocks: PriX speedup over CrypTen
<!-- generated:cpu_llm_speedup:begin -->
| Model | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **1.54x** | **3.26x** | **3.65x** | **2.33x** | **2.29x** |
| BertBase (1-Block, seq=16) | **2.57x** | **4.45x** | **5.47x** | **4.07x** | **4.71x** |
| GPT2 (1-Block, seq=16) | **4.08x** | **5.66x** | **6.63x** | **5.57x** | **6.12x** |
| Llama (1-Block, d=768, seq=16) | **3.55x** | **4.98x** | **5.38x** | **4.32x** | **4.76x** |
| ModernBert (1-Block, seq=16) | **3.30x** | **4.14x** | **4.87x** | **3.88x** | **4.21x** |
<!-- generated:cpu_llm_speedup:end -->

#### Primitives and nonlinearities (ms, CrypTen / PriX)
<!-- generated:cpu_ops:begin -->
| Benchmark | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| Private MatMul (128x128) | 23.39 / 7.71 (**3.03x**) | 28.73 / 6.99 (**4.11x**) | 27.91 / 5.47 (**5.10x**) | 29.27 / 7.06 (**4.15x**) | 23.00 / 7.04 (**3.26x**) |
| Private Mul (128x128) | 4.68 / 4.83 (**0.97x**) | 7.63 / 3.70 (**2.06x**) | 10.25 / 3.54 (**2.90x**) | 9.87 / 5.31 (**1.86x**) | 5.21 / 3.70 (**1.41x**) |
| Comparison / ReLU (32x64) | 31.85 / 12.52 (**2.54x**) | 26.98 / 6.76 (**3.99x**) | 28.05 / 13.62 (**2.06x**) | 24.39 / 8.65 (**2.82x**) | 33.79 / 9.28 (**3.64x**) |
| Exp (32x64) | 17.49 / 14.23 (**1.23x**) | 65.25 / 23.83 (**2.74x**) | 17.41 / 10.05 (**1.73x**) | 54.63 / 17.19 (**3.18x**) | 6.62 / 1.78 (**3.72x**) |
| Reciprocal (32x64) | 84.18 / 67.68 (**1.24x**) | 118.36 / 40.42 (**2.93x**) | 125.13 / 42.48 (**2.95x**) | 54.59 / 11.17 (**4.89x**) | 15.27 / 1.84 (**8.31x**) |
| InvSqrt (32x64) | 61.61 / 42.25 (**1.46x**) | 80.66 / 27.22 (**2.96x**) | 72.14 / 32.33 (**2.23x**) | 98.04 / 22.25 (**4.41x**) | 12.78 / 1.63 (**7.83x**) |
| GeLU (32x64) | 99.95 / 94.32 (**1.06x**) | 59.69 / 17.24 (**3.46x**) | 73.07 / 16.91 (**4.32x**) | 29.65 / 7.37 (**4.03x**) | 6.85 / 5.55 (**1.23x**) |
| SiLU (32x64) | 78.15 / 59.42 (**1.32x**) | 56.59 / 21.18 (**2.67x**) | 78.37 / 20.11 (**3.90x**) | 28.16 / 6.90 (**4.08x**) | 7.59 / 2.93 (**2.59x**) |
| Softmax (4x32x32) | 372.66 / 220.91 (**1.69x**) | 422.02 / 153.77 (**2.74x**) | 252.02 / 87.37 (**2.88x**) | 358.12 / 141.21 (**2.54x**) | 15.63 / 4.67 (**3.35x**) |
| LayerNorm (4x32x64) | 67.55 / 40.53 (**1.67x**) | 77.46 / 24.79 (**3.12x**) | 100.27 / 29.37 (**3.41x**) | 68.58 / 16.00 (**4.29x**) | 18.44 / 1.80 (**10.27x**) |
<!-- generated:cpu_ops:end -->

### 1.3 GPU (NVIDIA Tesla T4, Google Colab)

<!-- generated:gpu_env:begin -->
cpu_model: `Intel(R) Xeon(R) CPU @ 2.00GHz` · gpu_model: `Tesla T4` · python_version: `3.13.16` · jax_version: `0.12.0.1` · torch_version: `2.11.0+cu130` · prix_commit: `f6194ab94d6347b7ce784816072504e080e1485a` · timestamp_utc: `2026-10-24T20:13:57Z` · loadavg_1m_at_start: `2.34326171875`
<!-- generated:gpu_env:end -->

#### Transformer blocks: latency (ms)
<!-- generated:gpu_llm_latency:begin -->
| Model | CrypTen · CrypTen approx. | PriX · CrypTen approx. | CrypTen · Piecewise poly | PriX · Piecewise poly | CrypTen · SHAFT (Fourier + ODE) | PriX · SHAFT (Fourier + ODE) | CrypTen · Curl LUTs | PriX · Curl LUTs | CrypTen · Fission | PriX · Fission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | 2025.75 | 1066.40 | 2350.60 | 1012.24 | 2015.59 | 668.24 | 2050.49 | 788.97 | 416.14 | 195.42 |
| BertBase (1-Block, seq=16) | 1753.74 | 880.04 | 1933.84 | 877.54 | 1742.98 | 751.37 | 1767.11 | 692.53 | 932.82 | 401.84 |
| GPT2 (1-Block, seq=16) | 1693.46 | 904.82 | 1911.50 | 869.61 | 1666.38 | 695.62 | 1720.13 | 690.03 | 874.10 | 414.40 |
| Llama (1-Block, d=768, seq=16) | 1686.48 | 860.40 | 1929.48 | 865.08 | 1748.71 | 697.39 | 1801.39 | 741.53 | 1019.88 | 469.27 |
| ModernBert (1-Block, seq=16) | 1494.21 | 771.50 | 1705.33 | 783.20 | 1507.36 | 584.68 | 1562.69 | 631.61 | 716.10 | 341.34 |
<!-- generated:gpu_llm_latency:end -->

#### Transformer blocks: PriX speedup over CrypTen
<!-- generated:gpu_llm_speedup:begin -->
| Model | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| BertTiny (2-Block, seq=16) | **1.90x** | **2.32x** | **3.02x** | **2.60x** | **2.13x** |
| BertBase (1-Block, seq=16) | **1.99x** | **2.20x** | **2.32x** | **2.55x** | **2.32x** |
| GPT2 (1-Block, seq=16) | **1.87x** | **2.20x** | **2.40x** | **2.49x** | **2.11x** |
| Llama (1-Block, d=768, seq=16) | **1.96x** | **2.23x** | **2.51x** | **2.43x** | **2.17x** |
| ModernBert (1-Block, seq=16) | **1.94x** | **2.18x** | **2.58x** | **2.47x** | **2.10x** |
<!-- generated:gpu_llm_speedup:end -->

#### Primitives and nonlinearities (ms, CrypTen / PriX)
<!-- generated:gpu_ops:begin -->
| Benchmark | CrypTen approx. | Piecewise poly | SHAFT (Fourier + ODE) | Curl LUTs | Fission |
|---|---:|---:|---:|---:|---:|
| Private MatMul (128x128) | 12.58 / 5.50 (**2.29x**) | 16.42 / 3.89 (**4.22x**) | 16.72 / 4.00 (**4.18x**) | 16.83 / 8.30 (**2.03x**) | 15.27 / 4.44 (**3.44x**) |
| Private Mul (128x128) | 4.61 / 3.64 (**1.27x**) | 9.89 / 3.44 (**2.87x**) | 10.03 / 6.36 (**1.58x**) | 9.48 / 6.80 (**1.39x**) | 4.76 / 4.66 (**1.02x**) |
| Comparison / ReLU (32x64) | 50.91 / 20.02 (**2.54x**) | 47.59 / 18.19 (**2.62x**) | 48.09 / 17.84 (**2.70x**) | 51.17 / 18.77 (**2.73x**) | 50.32 / 16.99 (**2.96x**) |
| Exp (32x64) | 30.94 / 21.18 (**1.46x**) | 127.25 / 36.12 (**3.52x**) | 28.69 / 21.20 (**1.35x**) | 68.97 / 39.11 (**1.76x**) | 7.57 / 2.05 (**3.69x**) |
| Reciprocal (32x64) | 141.16 / 79.62 (**1.77x**) | 193.31 / 80.54 (**2.40x**) | 188.26 / 77.65 (**2.42x**) | 62.47 / 15.71 (**3.98x**) | 18.63 / 2.09 (**8.93x**) |
| InvSqrt (32x64) | 96.69 / 46.95 (**2.06x**) | 132.69 / 49.69 (**2.67x**) | 132.08 / 49.10 (**2.69x**) | 149.15 / 16.40 (**9.10x**) | 18.42 / 2.06 (**8.96x**) |
| GeLU (32x64) | 152.80 / 98.46 (**1.55x**) | 86.97 / 41.21 (**2.11x**) | 110.17 / 32.73 (**3.37x**) | 26.95 / 16.41 (**1.64x**) | 8.67 / 2.18 (**3.98x**) |
| SiLU (32x64) | 139.03 / 63.97 (**2.17x**) | 84.09 / 42.77 (**1.97x**) | 112.45 / 34.77 (**3.23x**) | 28.91 / 15.44 (**1.87x**) | 8.31 / 1.74 (**4.77x**) |
| Softmax (4x32x32) | 571.75 / 290.44 (**1.97x**) | 678.85 / 323.40 (**2.10x**) | 417.51 / 130.10 (**3.21x**) | 553.46 / 249.28 (**2.22x**) | 7.68 / 1.68 (**4.57x**) |
| LayerNorm (4x32x64) | 101.50 / 57.47 (**1.77x**) | 156.68 / 59.39 (**2.64x**) | 144.26 / 59.70 (**2.42x**) | 176.03 / 30.35 (**5.80x**) | 9.50 / 2.72 (**3.49x**) |
<!-- generated:gpu_ops:end -->

## 2. FSS and wavelet modes: default, Wave, SIGMA, Curl, SHAFT

Two parties, CPU, $f=16$, `BERT-base` shapes ($L=128$). Online bytes and preprocessing key bytes are **per party**; key bytes count seed correction words at $\lambda=128$ (16 B per DPF level, as SIGMA/Orca do) so they are comparable with SIGMA's Table 2, which PriX's `sigma` mode reproduces within 4 %. `sigma_ext` is SIGMA with the default preset's range-extended `inv_sqrt`, so that the block-error gap between `sigma` and `default` can be attributed to LayerNorm input range rather than protocol. Script: `benchmarks/benchmark_fss_modes.py`.

> [!WARNING]
> All latencies are **online-phase** numbers with a **simulated dealer** (`provider=TFP`): both parties expand the same PRG seeds, so FSS correction words, Beaver tuples and masks are derived locally and **no key material is transmitted**. This is the convention of CrypTen's `TrustedFirstParty` and of the online columns in the SIGMA/Wave papers, but it is **not a secure deployment**, and SIGMA's published *GPU* latencies include host→device key transfer; do not compare device-local PriX timings against them. The key column is what a real dealer would have to send (`prix.provider.get_preproc_stats()`), and it exceeds the online traffic by 8–14× for every FSS scheme.

### 2.1 `BERT-base`, per party
<!-- generated:fss_bert_base:begin -->
| `mode` | Nonlinearities | Rounds | Online | Key (λ=128) | CPU time | Block MAE |
|---|---|---:|---:|---:|---:|---:|
| `default` | Wave bior(5,3) FSS tables + SIGMA Truncate-Reduce + `[EGK+20]` | 828 | 0.83 GB | 8.1 GB | 22.7 s | 0.0007 |
| `wave` | 2-round Wave (Grotto DPF + DCF) | **768** | **0.78 GB** | 11.0 GB | 53.5 s | 0.0028 |
| `wave`, `lut_indexing="sigma"` | Wave tables, SIGMA Truncate-Reduce indexing (3 rounds) | 828 | 0.83 GB | 9.3 GB | 23.4 s | 0.0028 |
| `wave`, `lut_indexing="wave_prob"` | Wave tables, probabilistic indexing | **768** | **0.78 GB** | 6.6 GB | 19.0 s | 0.0191 |
| `sigma` | SIGMA 0th-order Haar FSS LUTs | 1,032 | 0.98 GB | 13.9 GB | 62.1 s | 0.0017 |
| `sigma` + wide `inv_sqrt` | SIGMA with the default preset's LayerNorm domain | 936 | 0.98 GB | 13.9 GB | 66.6 s | **0.0006** |
| `curl` | Curl wavelet LUTs (one-hot) + `[EGK+20]` | 1,668 | 3.12 GB | **6.4 GB** | **17.9 s** | 0.0192 |
| `shaft` | SHAFT Fourier activations + ODE Softmax | 1,200 | 5.51 GB | 9.1 GB | 23.6 s | 0.0015 |
<!-- generated:fss_bert_base:end -->

### 2.2 Sublayers and models
<!-- generated:fss_sublayers:begin -->
| Benchmark | Metric | **PriX default** | **Wave** | **Wave + SIGMA TR** | **Wave-Prob** | **SIGMA** | **SIGMA + ext. LN** | **Curl** | **SHAFT** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GeLU `(1,128,3072)`** | rounds / online 1P | 3 / 18.9 MB | 2 / 15.7 MB | 3 / 18.9 MB | 2 / 15.7 MB | 6 / 22.1 MB | 6 / 22.1 MB | 4 / 15.7 MB | 11 / 213.9 MB |
| | key 1P (λ=128) / time / MAE | 208 MB / 0.60 s / 4.5e-04 | 302 MB / 1.60 s / 1.9e-03 | 211 MB / 0.43 s / 1.9e-03 | 60 MB / 0.24 s / 9.8e-03 | 585 MB / 2.68 s / 3.8e-04 | 585 MB / 2.66 s / 3.8e-04 | 119 MB / 0.13 s / 9.7e-03 | 403 MB / 0.50 s / 9.8e-04 |
| **Softmax `(1,12,128,128)`** | rounds / online 1P | 25 / 19.0 MB | 23 / 17.4 MB | 25 / 19.0 MB | 23 / 17.4 MB | 31 / 28.4 MB | 31 / 28.4 MB | 98 / 155.9 MB | 41 / 173.0 MB |
| | key 1P (λ=128) / time / MAE | 418 MB / 0.71 s / 6.9e-05 | 564 MB / 1.80 s / 6.9e-05 | 518 MB / 0.88 s / 6.9e-05 | 442 MB / 0.77 s / 2.8e-04 | 526 MB / 2.10 s / 1.1e-05 | 526 MB / 2.22 s / 1.1e-05 | 312 MB / 0.64 s / 2.8e-04 | 271 MB / 0.64 s / 3.0e-04 |
| **LayerNorm `(1,128,768)`** | rounds / online 1P | 12 / 4.7 MB | 11 / 4.7 MB | 12 / 4.7 MB | 11 / 4.7 MB | 16 / 4.7 MB | 12 / 4.7 MB | 10 / 4.7 MB | 20 / 2.4 MB |
| | key 1P (λ=128) / time / MAE | 8 MB / 0.10 s / 4.3e-05 | 8 MB / 0.34 s / 4.3e-05 | 8 MB / 0.09 s / 4.3e-05 | 7 MB / 0.08 s / 1.7e-04 | 7 MB / 0.07 s / 9.3e-04 | 8 MB / 0.11 s / 4.3e-05 | 8 MB / 0.06 s / 4.3e-05 | 5 MB / 0.05 s / 9.0e-04 |
| **`BERT-base` (12L, L=128)** | rounds / online 1P | 828 / 0.833 GB | 768 / 0.776 GB | 828 / 0.833 GB | 768 / 0.776 GB | 1,032 / 0.984 GB | 936 / 0.984 GB | 1,668 / 3.117 GB | 1,200 / 5.512 GB |
| | key 1P (λ=128) / time / block MAE | 8.07 GB / 22.7 s / 0.0007 | 10.96 GB / 53.5 s / 0.0028 | 9.31 GB / 23.4 s / 0.0028 | 6.58 GB / 19.0 s / 0.0191 | 13.89 GB / 62.1 s / 0.0017 | 13.90 GB / 66.6 s / 0.0006 | 6.43 GB / 17.9 s / 0.0192 | 9.12 GB / 23.6 s / 0.0015 |
| **`BERT-large` (24L, L=128)** | rounds / online 1P | 1,656 / 2.220 GB | 1,536 / 2.068 GB | 1,656 / 2.220 GB | 1,536 / 2.068 GB | 2,064 / 2.624 GB | 1,872 / 2.625 GB | 3,336 / 8.916 GB | 2,400 / 15.302 GB |
| | key 1P (λ=128) / time / block MAE | 21.50 GB / 67.0 s / 0.0007 | 29.21 GB / 135.9 s / 0.0028 | 24.81 GB / 61.7 s / 0.0028 | 17.54 GB / 50.5 s / 0.0190 | 37.03 GB / 183.1 s / 0.0019 | 37.04 GB / 200.8 s / 0.0006 | 17.74 GB / 50.1 s / 0.0193 | 24.92 GB / 64.1 s / 0.0015 |
| **`GPT-2` (12L, L=64)** | rounds / online 1P | 804 / 0.345 GB | 744 / 0.322 GB | 804 / 0.345 GB | 744 / 0.322 GB | 1,008 / 0.393 GB | 912 / 0.393 GB | 1,656 / 1.391 GB | 1,200 / 2.572 GB |
| | key 1P (λ=128) / time / block MAE | 2.76 GB / 10.9 s / 0.0007 | 3.77 GB / 23.2 s / 0.0028 | 3.08 GB / 13.1 s / 0.0028 | 1.94 GB / 8.8 s / 0.0189 | 5.34 GB / 26.3 s / 0.0017 | 5.35 GB / 30.7 s / 0.0006 | 2.56 GB / 11.3 s / 0.0194 | 4.08 GB / 11.4 s / 0.0015 |
| **`GPT-2` (12L, L=128)** | rounds / online 1P | 828 / 0.833 GB | 768 / 0.776 GB | 828 / 0.833 GB | 768 / 0.776 GB | 1,032 / 0.984 GB | 936 / 0.984 GB | 1,668 / 3.117 GB | 1,200 / 5.512 GB |
| | key 1P (λ=128) / time / block MAE | 8.07 GB / 22.6 s / 0.0007 | 10.96 GB / 52.3 s / 0.0028 | 9.31 GB / 21.9 s / 0.0028 | 6.58 GB / 18.0 s / 0.0191 | 13.89 GB / 63.7 s / 0.0016 | 13.90 GB / 66.4 s / 0.0006 | 6.43 GB / 18.1 s / 0.0194 | 9.12 GB / 20.7 s / 0.0015 |
<!-- generated:fss_sublayers:end -->

> [!IMPORTANT]
> **Input range matters more than table resolution.** Before v0.2.1 every LUT mode (Curl, SIGMA, Wave) had a Transformer-block MAE of 0.045–0.051 regardless of how accurate its primitives were: the softmax denominator $\sum_j e^{x_j-\max} \in [1, L]$ exceeded the reciprocal table's $[0, 64)$ domain whenever attention was near-uniform, and LayerNorm's `inv_sqrt` table stopped at variance 16. PriX now sizes the inverse table from the softmax length (as SIGMA §5.2.3 does) and the `default`/`wave` presets use a range-extended `inv_sqrt` (variances in $[2^{-8}, 2^{16})$). The `sigma_ext` column shows the same fix applied to SIGMA: its block MAE drops from 0.0017 to 0.0006, i.e. SIGMA's and Wave's protocols are equally accurate once both see in-range inputs. See Section 6.3 of the paper.

### 2.3 FSS modes on a GPU (Tesla T4)

Rounds, bytes, key sizes and errors are identical to the CPU session (the protocols do not change); only the local compute does. With both parties sharing one T4 and PriX's FP64-limb matrix multiplication (the T4 has 1/32-rate FP64), every FSS variant is **slower on the T4 than on the 24-vCPU CPU host**; the ranking between variants is the same. SIGMA's published GPU figures include key transfer and use AES-NI/CUTLASS kernels, so they are not comparable with this column (see the warning above).

<!-- generated:fss_gpu:begin -->
GPU session: `Tesla T4` (Google Colab), JAX `0.12.0.1`, commit `5ee61bc`, 3 repeats, both parties on one GPU. CPU column: the session of §2.1/§2.2.

| Model | Metric | **Wave** | **Wave + SIGMA TR** | **Wave-Prob** | **SIGMA** | **Curl** | **SHAFT** |
|---|---|---:|---:|---:|---:|---:|---:|
| **`BERT-base` (12L)** | time, CPU (AMD EPYC 7B12, 24 vCPU) | 53.5 s | 23.4 s | 19.0 s | 62.1 s | 17.9 s | 23.6 s |
| | time, GPU (Tesla T4) | 167.5 s | 61.9 s | 59.1 s | 297.1 s | 75.4 s | 80.0 s |
| | block MAE (GPU) | 0.0028 | 0.0028 | 0.0192 | 0.0016 | 0.0191 | 0.0015 |
| **`BERT-large` (24L)** | time, CPU (AMD EPYC 7B12, 24 vCPU) | 135.9 s | 61.7 s | 50.5 s | 183.1 s | 50.1 s | 64.1 s |
| | time, GPU (Tesla T4) | 456.3 s | 188.1 s | 177.5 s | 844.5 s | 254.3 s | 265.0 s |
| | block MAE (GPU) | 0.0028 | 0.0028 | 0.0193 | 0.0018 | 0.0188 | 0.0015 |
| **`GPT-2` (12L, L=64)** | time, CPU (AMD EPYC 7B12, 24 vCPU) | 23.2 s | 13.1 s | 8.8 s | 26.3 s | 11.3 s | 11.4 s |
| | time, GPU (Tesla T4) | 74.0 s | 35.5 s | 30.2 s | 114.2 s | 46.2 s | 50.9 s |
| | block MAE (GPU) | 0.0028 | 0.0028 | 0.0193 | 0.0019 | 0.0188 | 0.0015 |
<!-- generated:fss_gpu:end -->

### 2.4 Deterministic vs. probabilistic truncation (2-round Wave)

`benchmarks/benchmark_fss_modes.py --ablation` runs the four combinations of `lut_indexing ∈ {wave, wave_prob}` × `truncation ∈ {egk, crypten}`; the paper's Section 6.4 discusses them. In short: probabilistic LUT indexing (`wave_prob`) is ~3× faster locally and needs 40 % less key but raises block MAE from 0.0028 to 0.019 (Curl's level); local product truncation (`crypten`) saves 25 % of the rounds with no measurable accuracy change on these blocks, but is not a secure protocol in general (Section 6.4 of the paper explains the wrap-around failure modes).
