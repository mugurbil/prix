<p align="center"><img width="35%" src="https://raw.githubusercontent.com/mugurbil/prix/main/prix.png" alt="PriX logo" /></p>

[![PyPI version](https://img.shields.io/pypi/v/prix-jax.svg)](https://pypi.org/project/prix-jax/)
[![Tests](https://github.com/mugurbil/prix/actions/workflows/tests.yml/badge.svg)](https://github.com/mugurbil/prix/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugurbil/prix/blob/main/examples/01_quickstart_mpc_basics.ipynb)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mugurbil/prix/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mugurbil/prix/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

**Fast private Transformer inference in JAX: every major MPC method, one API, CPU or GPU.**

PriX lets a model owner and a data owner run BERT, GPT-2 or Llama together without either side seeing the other's weights or inputs, using secure multi-party computation (MPC).

<p align="center"><img src="https://raw.githubusercontent.com/mugurbil/prix/main/docs/figures/llm_latency.png" alt="GPT-2 block latency, CrypTen vs PriX, CPU and GPU, five modes" width="95%"/></p>
<!-- generated:hero_caption:begin -->
<p align="center"><sub>One GPT-2 block, 2 parties, same protocols on both sides: up to 6.6× faster on CPU and 2.5× on a T4 GPU. <a href="docs/benchmarks.md">More benchmarks</a></sub></p>
<!-- generated:hero_caption:end -->

## Why PriX?

- **Fast.** MPC protocols compile into fused XLA kernels, with exact 64-bit ring matmuls on FP64 hardware and batched peer-to-peer messaging. No per-operation Python overhead.
- **Every method, one API.** [Wave](https://doi.org/10.56553/popets-2025-0083), [SIGMA](https://doi.org/10.56553/popets-2024-0107), [Curl](https://eprint.iacr.org/2024/1127), [PUMA](https://arxiv.org/abs/2307.12533)-style polynomials, [SHAFT](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/), [CrypTen](https://arxiv.org/abs/2109.00984) and [Fission](https://eprint.iacr.org/2025/653) are one `mode=` argument away, so comparing papers fairly is a `for` loop, not a port.
- **Strong default.** The default preset matches SIGMA's accuracy on BERT-base with 20% fewer rounds and 42% smaller keys ([details](docs/benchmarks.md#2-fss-and-wavelet-modes-default-wave-sigma-curl-shaft)).
- **Familiar.** A CrypTen-style API (`prix.array`, `prix.nn`, `@prix.mpc.run_multiprocess`), so existing CrypTen code ports with minimal changes ([migration guide](docs/quickstart.md)).

```python
import prix

@prix.mpc.run_multiprocess(world_size=2)
def demo():
    x = prix.array([[1.0, -0.5, 2.0]], src=0)                        # party 0's input
    w = prix.array([[0.5, 1.0], [-1.0, 0.25], [0.8, -0.5]], src=1)  # party 1's weights
    return (x @ w).gelu().softmax(dim=-1).reconstruct()

print(demo())
```

> [!WARNING]
> **Research prototype, not for real secrets.** Security is semi-honest, the default dealer is simulated and the FSS PRG is not cryptographic. See [Security](docs/security.md).

## Installation

```bash
pip install prix-jax    # or: git clone https://github.com/mugurbil/prix.git && cd prix && pip install -e .
```
Python 3.10–3.13; JAX on CPU or CUDA (`pip install "jax[cuda12]"`).

## Quickstart

**Switch methods without touching model code**
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

Every approximation, truncation and bit-width is tunable through `prix.cfg`; see [Modes & configuration](docs/modes_and_configuration.md).

## Documentation

| | |
|---|---|
| [Paper (PDF)](paper/prix_arxiv.pdf) · [LaTeX](paper/main.tex) | *PriX: Private Accelerated Transformers via Secure Multi-Party Computation in JAX* |
| [Quickstart & CrypTen migration](docs/quickstart.md) | Side-by-side cheat sheet, arithmetic vs. binary sharing, `TFP` vs. `TTP`. |
| [Modes & configuration](docs/modes_and_configuration.md) | Every preset, every `prix.cfg` knob, valid input ranges per function. |
| [Benchmarks](docs/benchmarks.md) | CPU/GPU speedups, transport ablation, FSS protocol study with key sizes. Reproduce with `benchmarks/run_all.sh`; every number traces to a results file ([provenance](docs/provenance.md)). |
| [Transports & multi-host](docs/transports.md) | Peer-to-peer TCP (optional mTLS), shared memory, `gloo`, WAN emulation. |
| [Security](docs/security.md) | Threat model, dealer, PRG and input-range caveats. |
| [API reference](docs/api_reference.md) | `prix.*`, `prix.Array`, `prix.nn`. |
| Colab notebooks | [01 MPC basics](examples/01_quickstart_mpc_basics.ipynb) · [02 Nonlinearity ablations](examples/02_approximations_and_ablations.ipynb) · [03 Transformers & LLMs](examples/03_private_transformers_and_llms.ipynb) · [04 PriX vs. CrypTen on GPU](examples/prix_vs_curl_gpu_colab.ipynb) · [05 FSS modes on GPU](examples/prix_fss_wave_sigma_gpu_colab.ipynb) |

Tests: `JAX_PLATFORMS=cpu python -m pytest tests -q`.

## Citing PriX

```bibtex
@misc{ugurbil2026prix,
  author       = {Mehmet Ugurbil},
  title        = {{PriX}: Private Accelerated Transformers via Secure Multi-Party Computation in {JAX}},
  howpublished = {Cryptology ePrint Archive},
  year         = {2026},
  url          = {https://github.com/mugurbil/prix}
}
```

PriX implements protocols from CrypTen (Knott et al., NeurIPS 2021), SIGMA (Gupta et al., PoPETs 2024), Wave (Reis et al., PoPETs 2025), Curl (Santos et al., CAMLIS 2024), PUMA (Dong et al., 2023), SHAFT (Kei & Chow, NDSS 2025), Fission (Ugurbil et al., IEEE CNS 2025) and TruncPr (Escudero et al., CRYPTO 2020); full citations are in the [paper](paper/prix_arxiv.pdf).

Research prototype; not production-ready. Use at your own risk.
