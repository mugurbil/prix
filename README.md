<p align="center"><img width="35%" src="prix.png" alt="PriX logo" /></p>

[![PyPI version](https://img.shields.io/pypi/v/prix-jax.svg)](https://pypi.org/project/prix-jax/)
[![Tests](https://github.com/mugurbil/prix/actions/workflows/tests.yml/badge.svg)](https://github.com/mugurbil/prix/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/01_quickstart_mpc_basics.ipynb)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**PriX** is a JAX/XLA framework for **private Transformer inference with secure multi-party computation (MPC)**. It is a **drop-in alternative to [CrypTen](https://github.com/facebookresearch/CrypTen)** — same API, 2–4× faster, maintained on current Python/JAX — that **also subsumes [SIGMA](https://doi.org/10.56553/popets-2024-0107)/[Wave](https://doi.org/10.56553/popets-2025-0083)-style Function Secret Sharing (FSS)**: DPF/DCF comparisons, faithful truncation and wavelet look-up tables, all compiled to fused XLA kernels on CPU and GPU.

```python
import prix

@prix.mpc.run_multiprocess(world_size=2)          # default mode: Wave FSS tables + SIGMA truncation
def demo():
    x = prix.array([[1.0, -0.5, 2.0]], src=0)      # party 0's input
    w = prix.array([[0.5, 1.0], [-1.0, 0.25], [0.8, -0.5]], src=1)  # party 1's weights
    return (x @ w).gelu().softmax(dim=-1).reconstruct()

print(demo())
```

Eight nonlinearity modes share one API and are selected with a single argument, `mode="default" | "wave" | "sigma" | "curl" | "poly" | "shaft" | "crypten" | "fission"` (PriX's own default, then [Wave](https://doi.org/10.56553/popets-2025-0083), [SIGMA](https://doi.org/10.56553/popets-2024-0107), [Curl](https://eprint.iacr.org/2024/1127), [PUMA](https://arxiv.org/abs/2307.12533), [SHAFT](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/), [CrypTen](https://arxiv.org/abs/2109.00984), [Fission](https://eprint.iacr.org/2025/653)), so a cross-paper ablation is a `for` loop.

> [!WARNING]
> **Research prototype — not suitable for protecting real secrets.** Security is semi-honest. The default `TFP` provider is a *simulated* dealer (both parties expand the same seeds, so no key material is ever transferred and party 0 could learn every secret — as in CrypTen); use `TTP` for a non-colluding dealer. FSS PRG trees are expanded with `SplitMix64` from 62-bit seeds (`cfg.mpc.prng="siphash24"` for a keyed PRF), which is *not* a cryptographic PRG at 128-bit security. See [Security](#security-and-what-prix-is-not).

---

## What you get

`BERT-base` (12 layers, $L=128$), 2 parties, one host, CPU, $f=16$. Online rounds/bytes and preprocessing key bytes are per party; details, GPU numbers and primitives in [docs/benchmarks.md](docs/benchmarks.md) and Section 6 of the [paper](paper/prix_arxiv.pdf).

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
| `crypten` / `poly` / `fission` | CrypTen Newton/limit · PUMA polynomials · hybrid evaluator | see [benchmarks](docs/benchmarks.md) | | | | |
<!-- generated:fss_bert_base:end -->

- **vs. CrypTen**: identical API (`prix.cryptensor`, `@prix.mpc.run_multiprocess`, `prix.nn`), **1.5–6.6× faster Transformer blocks on CPU (geomean 4.0×) and 1.9–3.1× on GPU (geomean 2.3×)**, with the same protocols ([details](docs/benchmarks.md#prix-jax-vs-crypten-pytorch)).
- **vs. SIGMA**: PriX's `sigma` mode reproduces SIGMA's communication and key sizes within 4% (Tables 2/5 of their paper); the `default` preset then needs **20% fewer rounds, 15% less online traffic, 42% smaller keys and ~2.7× less CPU time at equal accuracy** (block MAE 0.0007 vs. 0.0006 once SIGMA's LayerNorm table covers the same variance range — SIGMA's published configuration gives 0.0017). SIGMA's C++/CUDA build remains faster per PRG call (AES-NI; PriX is Python/XLA) and ships a cryptographic PRG.
- **The protocol study** (paper §6) is reproducible with one script: `python benchmarks/benchmark_fss_modes.py [--device cuda]`, or the [GPU Colab](examples/prix_fss_wave_sigma_gpu_colab.ipynb).

---

## Installation

```bash
pip install prix-jax            # PyPI
# or
git clone https://github.com/mugurbil/prix.git && cd prix && pip install -e .
```
Python 3.10–3.13; JAX on CPU or CUDA (`pip install "jax[cuda12]"`).

---

## Quickstart

**Switch paradigms without touching model code**
```python
def eval_gelu():
    return prix.array([-2.0, -0.5, 0.0, 0.5, 2.0]).gelu().reconstruct()

for mode in ["default", "wave", "sigma", "curl", "poly", "shaft", "crypten", "fission"]:
    print(f"[{mode:>8}]", prix.run(eval_gelu, world_size=2, mode=mode).round(4))
```

**Private Transformers with `prix.nn`**
```python
import prix.nn as nn

class Block(nn.Module):
    def __init__(self, d=768, h=12, ffn=3072):
        super().__init__()
        self.attn, self.ln1, self.ln2 = nn.MultiheadAttention(d, h), nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc1, self.act, self.fc2 = nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d)
    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        return self.ln2(x + self.fc2(self.act(self.fc1(x))))

@prix.mpc.run_multiprocess(world_size=2)
def run():
    model = Block().encrypt(src=0)         # model owner = party 0
    x = prix.randn(1, 128, 768)             # secret-shared input
    return model(x).reconstruct().shape
```

**Mix methods and truncations via `prix.cfg`**
```python
prix.cfg.functions.gelu.bior_size_bits = 6      # 64-knot GeLU spline (more accurate, more PRG calls)
prix.cfg.encoder.truncation = "egk"             # product truncation: egk | crypten | fission
prix.cfg.encoder.lut_indexing = "sigma"         # LUT bin indexing: wave | wave_prob | sigma | egk | crypten
prix.cfg.mpc.fss_bits = 48                      # effective bit-width of FSS comparisons
```
`truncation` rescales every product by $2^{-f}$; `lut_indexing` maps a secret LUT *input* to its bin ($\gg \mathfrak j$) *before* the lookup. They are separate knobs because a ±1 error is 1 ULP in the first case and a jump to the neighbouring wavelet knot in the second, and because in FSS modes the indexing is fused into the LUT protocol. See [Modes & Configuration](docs/modes_and_configuration.md).

---

## Documentation

| | |
|---|---|
| [Paper (PDF)](paper/prix_arxiv.pdf) · [LaTeX](paper/main.tex) | *PriX: Private Accelerated Transformers via Secure Multi-Party Computation in JAX* — architecture, CrypTen comparison, and the FSS/wavelet protocol study (§6). |
| [Quickstart & CrypTen migration](docs/quickstart.md) | Side-by-side cheat sheet, arithmetic vs. binary sharing, `TFP` vs. `TTP`. |
| [Modes & configuration](docs/modes_and_configuration.md) | Every preset, every `prix.cfg` knob, valid input ranges per function. |
| [Benchmarks](docs/benchmarks.md) | CPU/GPU tables and charts vs. CrypTen, transport ablation, FSS-mode comparison with key sizes. |
| [Transports & multi-host](docs/transports.md) | Peer-to-peer TCP (optional mTLS), shared memory, `gloo`, `TTP`/`fission` helpers, WAN emulation. |
| [API reference](docs/api_reference.md) | `prix.*`, `prix.Array`, `prix.nn`. |
| Colab notebooks | [01 MPC basics](examples/01_quickstart_mpc_basics.ipynb) · [02 Nonlinearity ablations](examples/02_approximations_and_ablations.ipynb) · [03 Transformers & LLMs](examples/03_private_transformers_and_llms.ipynb) · [04 PriX vs. CrypTen on GPU](examples/prix_vs_curl_gpu_colab.ipynb) · [05 FSS modes on GPU](examples/prix_fss_wave_sigma_gpu_colab.ipynb) |

Tests: `JAX_PLATFORMS=cpu python -m pytest tests -q`.

---

## Security, and what PriX is not

- **Threat model.** Semi-honest, dishonest-majority 2PC (and $N$-party additive sharing for the CrypTen-style modes) in the preprocessing model. `cfg.mpc.active_security` is a Beaver-consistency check, not malicious security.
- **Dealer.** `TFP` simulates the dealer: correlated randomness and FSS keys are expanded locally from shared seeds, so nothing is transferred and the measured latencies are *online-phase* numbers. `prix.provider.get_preproc_stats()` reports the bytes a real dealer would have to send; Section 6.2 of the paper tabulates them. SIGMA's published GPU latencies *include* key transfer; do not compare device-local PriX timings against them.
- **PRG.** FSS trees use `SplitMix64` with 62-bit seeds for speed; this is below 128-bit security and not a standard PRG assumption. A production build needs AES/ChaCha in `prix/fss.py`'s PRG.
- **Input ranges.** LUT modes are exact only on their configured domains; the `default` preset sizes the softmax reciprocal from the sequence length and uses a range-extended `inv_sqrt` (variances $[2^{-8}, 2^{16})$), but every other function's range is documented in [Modes & Configuration](docs/modes_and_configuration.md#5-accuracy-and-valid-input-ranges) and is **not** checked at run time.
- **Transport.** Parties talk over peer-to-peer TCP with a pickle-free wire format; enable mutual TLS for anything beyond experiments.

---

## References

- **CrypTen** — Knott et al., *CrypTen: Secure Multi-Party Computation Meets Machine Learning*, NeurIPS 2021 ([paper](https://arxiv.org/abs/2109.00984), [code](https://github.com/facebookresearch/CrypTen)).
- **SIGMA** — Gupta et al., *SIGMA: Secure GPT Inference with Function Secret Sharing*, PoPETs 2024 ([paper](https://doi.org/10.56553/popets-2024-0107), [code](https://github.com/mpc-msri/EzPC)).
- **Wave** — Reis, Ugurbil, Wagh, Henry, de Vega, *Wave Hello to Privacy: Efficient Mixed-Mode MPC using Wavelet Transforms*, PoPETs 2025 ([paper](https://doi.org/10.56553/popets-2025-0083)).
- **Curl** — Santos, Mouris, Ugurbil, Jarecki, Reis, Sengupta, de Vega, *Curl: Private LLMs through Wavelet-Encoded Look-Up Tables*, CAMLIS 2024 ([ePrint](https://eprint.iacr.org/2024/1127), [code](https://github.com/jimouris/curl)).
- **PUMA** — Dong et al., *PUMA: Secure Inference of LLaMA-7B in Five Minutes*, 2023 ([arXiv](https://arxiv.org/abs/2307.12533), [code](https://github.com/secretflow/spu)).
- **SHAFT** — Kei, Chow, *SHAFT: Secure, Handy, Accurate, and Fast Transformer Inference*, NDSS 2025 ([paper](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/), [code](https://github.com/andeskyl/SHAFT)).
- **Fission** — Ugurbil, Mouris, Santos, Cabrero-Holgueras, de Vega, Sengupta, *Fission: Distributed Privacy-Preserving Large Language Model Inference*, IEEE CNS 2025 ([ePrint](https://eprint.iacr.org/2025/653)).
- **TruncPr** — Escudero, Ghosh, Keller, Rachuri, Scholl, *Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits*, CRYPTO 2020 (`[EGK+20]`).

### Citing PriX

```bibtex
@misc{ugurbil2026prix,
  author       = {Mehmet Ugurbil},
  title        = {{PriX}: Private Accelerated Transformers via Secure Multi-Party Computation in {JAX}},
  howpublished = {Cryptology ePrint Archive},
  year         = {2026},
  url          = {https://github.com/mugurbil/prix}
}
```

## Disclaimer
Research prototype; not production-ready. Use at your own risk.
