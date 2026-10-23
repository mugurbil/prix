# MPC Approximation Modes & Configuration Reference

PriX supports seven nonlinear approximation paradigms out of the box, selectable via `mode="curl" | "wave" | "sigma" | "poly" | "shaft" | "crypten" | "fission"` in `prix.run`, `@prix.mpc.run_multiprocess`, `prix.init`, or `prix.set_mode`.

---

## 1. Overview of the Seven Modes

| Mode Name | Config File | Truncation | Nonlinear Approximations | Reference |
|---|---|---|---|---|
| **`"default"`** *(what you get with no `mode`)* | [`configs/default.yaml`](../configs/default.yaml) | `[EGK+20]` for products (`truncation="egk"`) + SIGMA's 1-round faithful Truncate-Reduce for LUT indexing (`lut_indexing="sigma"`); both are secure 2PC protocols (no local truncation) | Wave's $\text{bior}(5,3)$ & Haar FSS Wavelet LUTs (identical tables and accuracy to `"wave"`), evaluated as a $J$-level DPF after the single-path TR DCF: 3 rounds per LUT, $2\times$–$6\times$ less local compute and $15\%$–$30\%$ smaller keys than 2-round `"wave"` | Wave (PoPETs 2025) + SIGMA (PoPETs 2024) |
| **`"curl"`** | [`configs/curl.yaml`](../configs/curl.yaml) | `[EGK+20]` probabilistic (`egk`) | Biorthogonal-2.2 & Haar Wavelet LUTs (one-hot correlations) for `exp`, `log`, `reciprocal`, `sqrt`, `inv_sqrt`, `sigmoid`, `tanh`, `erf`, `gelu`, `silu` | [**Curl (CAMLIS 2024)**](https://eprint.iacr.org/2024/1127) |
| **`"wave"`** | [`configs/wave.yaml`](../configs/wave.yaml) | Deterministic DPF/DCF (`wave`) or probabilistic (`wave_prob`) for LUTs; `egk` or `crypten` for products | 2-round Function Secret Sharing (Grotto DPF segment parities + arithmetic DCF + generalized Beaver tuples) with $\text{bior}(5,3)$ & Haar Wavelet LUTs | [**Wave (PoPETs 2025)**](https://doi.org/10.56553/popets-2025-0083) |
| **`"sigma"`** | [`configs/sigma.yaml`](../configs/sigma.yaml) | Faithful Truncate-Reduce (`lut_indexing="sigma"`) + `[EGK+20]` (`egk`) | Faithful SIGMA two-phase FSS protocols (DPF sign/clip comparisons + $\Pi^{\text{Select}}$ mux + 1-round $\Pi^{\text{TR}}_{n,f}$ + 2-round Grotto DPF 0th-order LUTs, two-table $2^8\times 2^8$ `nexp`, tailored `inv_sqrt`) | [**SIGMA (PoPETs 2024)**](https://doi.org/10.56553/popets-2024-0107) |
| **`"poly"`** | [`configs/poly.yaml`](../configs/poly.yaml) | CrypTen (`crypten`) | Piecewise low-degree polynomials for `gelu`, `silu`, `exp`, `sigmoid`, `tanh`, `erf` (1 batched comparison + degree-4 polynomial, no LUTs) | [**PUMA (arXiv 2023)**](https://arxiv.org/abs/2307.12533), [**Nimbus (NeurIPS 2024)**](https://arxiv.org/abs/2411.15707) |
| **`"shaft"`** | [`configs/shaft.yaml`](../configs/shaft.yaml) | CrypTen (`crypten`) | 1-round trigonometric-triple Fourier series (`gelu`, `silu`, `erf`, `sigmoid`, `tanh`) + constant-round Forward-Euler ODE `softmax` | [**SHAFT (NDSS 2025)**](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/) |
| **`"crypten"`** | [`configs/approximations.yaml`](../configs/approximations.yaml) | CrypTen (`crypten`) | Limit `exp`, Newton–Raphson `reciprocal`/`sqrt`/`inv_sqrt`, Taylor `erf`, sigmoid-based `silu`, standard `softmax` | [**CrypTen (NeurIPS 2021)**](https://arxiv.org/abs/2109.00984) |
| **`"fission"`** | [`configs/fission.yaml`](../configs/fission.yaml) | Fission (`fission`) | Permute-and-reveal to a non-colluding plaintext evaluator party (`EvaluatorServer`, auto-spawned by `prix.run`) | [**Fission (IEEE CNS 2025)**](https://eprint.iacr.org/2025/653) |

---

## 2. Switching Modes in Code

### Whole-Experiment Mode Selection
Pass `mode=` to `@prix.mpc.run_multiprocess` or `prix.run`:
```python
import prix

@prix.mpc.run_multiprocess(world_size=2, mode="wave")
def run_wave():
    x = prix.array([-1.5, 0.0, 1.5])
    return x.gelu().reconstruct()
```

Or switch dynamically inside a process using `prix.set_mode`:
```python
prix.set_mode("curl")      # Curl Wavelet LUTs
prix.set_mode("wave")     # Wave 2-round FSS + bior(5,3)/Haar Wavelet LUTs
prix.set_mode("sigma")    # SIGMA 2-round FSS comparisons + Haar LUTs
prix.set_mode("poly")     # PUMA / Nimbus piecewise low-degree polynomials
prix.set_mode("shaft")    # SHAFT Fourier + ODE
prix.set_mode("crypten")  # CrypTen classic approximations
prix.set_mode("fission")  # Fission evaluator
```

> [!WARNING]
> Fission mode needs an evaluator process, which is only started when the run itself is launched in Fission mode (`prix.run(fn, mode="fission")`, or `evaluator_size>=1`). Switching to `"fission"` with `prix.set_mode` inside a run that was started without an evaluator makes later nonlinearities fail.

---

## 3. Mixing & Customizing Methods via `prix.cfg`

Because `prix.cfg` deep-merges every preset onto `default.yaml`, you can freely mix methods from different papers (for example, combining **SHAFT's ODE Softmax** and **1-round Fourier GeLU** with **Curl's Wavelet LUT LayerNorm/InvSqrt** and **EGK truncation**):

```python
import prix

@prix.mpc.run_multiprocess(world_size=2, mode="curl")
def hybrid_curl_shaft():
    # Base mode is "default" (Wave bior(5,3) FSS LUTs + SIGMA TR indexing + EGK truncation);
    # override GeLU and Softmax to use SHAFT's Fourier & ODE approximations:
    with prix.cfg.temp_override({
        "functions.gelu.method": "fourier",
        "functions.gelu.fs_period": 8,
        "functions.gelu.fs_terms": 8,
        "functions.softmax.method": "ode",
        "functions.softmax.ode_iter_num": 16,
    }):
        x = prix.randn(2, 16, 64)
        return x.gelu().softmax(dim=-1).reconstruct()
```

---

## 4. Key Configuration Options (`prix.cfg`)

| Config Path | Options / Default | Description |
|---|---|---|
| `encoder.precision_bits` | `16` (int) | Fixed-point fractional precision bits ($2^{16} = 65536$). |
| `encoder.truncation` | `"egk"` \| `"crypten"` \| `"fission"` | Fixed-point rescaling ($\gg f$) after **every product** (`matmul`, `mul`, `square`). A $\pm 1$ error here is 1 ULP, so probabilistic protocols are acceptable. *(Formerly `encoder.trunc_method.prod`.)* |
| `encoder.lut_indexing` | `"egk"` \| `"wave"` \| `"wave_prob"` \| `"sigma"` \| `"crypten"` \| `"fission"` | How a secret LUT **input** is reduced to its bin index ($\gg \mathfrak{j}$, where $\mathfrak{j}$ depends on the table, not on $f$) *before* the lookup. Kept separate from `truncation` because a $\pm 1$ error here is a jump to the neighbouring wavelet knot (Curl's boundary error), and because in FSS modes it is fused into the LUT protocol itself rather than being a standalone primitive (`"wave"` = deterministic DPF/DCF; `"wave_prob"` = 0-round local split + $J$-level DPF; `"sigma"` = SIGMA's 1-round faithful $\Pi^{\text{TR}}_{n,f}$ (single-path DCF) followed by a $J$-level DPF LUT; usable with `mode="sigma"` (Haar tables) **and** with `mode="wave"`, where it gives Wave's exact $\text{bior}(5,3)$ accuracy at 3 rounds with $2\times$–$6\times$ less local compute than the 2-round Grotto variant). |
| `mpc.protocol` | `"beaver"` \| `"fss"` | Underlying comparison & LUT protocol (`"fss"` uses 2-round Function Secret Sharing in `sigma` and `wave` modes). |
| `mpc.provider` | `"TFP"` \| `"TTP"` | Correlated randomness dealer (`TrustedFirstParty` vs. `TrustedThirdParty`). |
| `functions.exp.method` | `"bior"` \| `"haar"` \| `"split"` \| `"limit"` \| `"poly"` \| `"fission"` | Exponential approximation (`iterations`, `bior_size_bits`, `lut_max_bits`, `neg_lut_size`). |
| `functions.reciprocal.method` | `"bior"` \| `"haar"` \| `"NR"` \| `"fission"` | Reciprocal $1/x$ (`nr_iters`, `bior_size_bits`, `lut_max_bits`). |
| `functions.inv_sqrt.method` | `"bior"` \| `"haar"` \| `"tailored_haar"` \| `"NR"` \| `"fission"` | Inverse square root $1/\sqrt{x}$ used in `LayerNorm` and `RMSNorm`. |
| `functions.gelu.method` | `"bior-lut-only"` \| `"haar-lut-only"` \| `"haar"` \| `"fourier"` \| `"erf"` \| `"poly"` \| `"fission"` | GeLU activation (`fs_period`, `fs_terms` for SHAFT Fourier; `bior_size_bits` for Curl/Wave LUT; `"haar"` for SIGMA two-phase ReLU + clip + LUT). |
| `functions.silu.method` | `"bior-lut-only"` \| `"haar-lut-only"` \| `"haar"` \| `"fourier"` \| `"sigmoid"` \| `"poly"` \| `"fission"` | SiLU / Swish activation (`fs_period`, `fs_terms` for SHAFT Fourier; `"haar"` for SIGMA two-phase ReLU + clip + LUT). |
| `functions.sigmoid_tanh.method` | `"bior"` \| `"fourier"` \| `"reciprocal"` \| `"chebyshev"` \| `"poly"` \| `"fission"` | Sigmoid and Tanh activations. |
| `functions.softmax.method` | `"approximation"` \| `"ode"` \| `"fission"` | Softmax (`"ode"` uses SHAFT Forward-Euler ODE with `ode_iter_num`, `ode_ub`, `ode_lb`). |
| `functions.layernorm.method` | `"approximation"` \| `"fission"` | LayerNorm / RMSNorm normalization method. |

---

## 5. Accuracy and Valid Input Ranges

Each mode's approximations are only accurate on a limited input range, and **inputs outside that range are not clamped or flagged: they silently return wrong values.** The table shows the measured error of each function as shipped (default config of each mode, 2 parties, 10 repeats, compared with exact float64 results). Each cell is *median absolute error / typical worst error within one run*; ⚠ marks runs where some output was off by more than 1.

| Function (tested inputs) | `crypten` | `poly` | `shaft` | `curl` | `fission` |
|---|---:|---:|---:|---:|---:|
| exp, x ≤ 0 ([−8, 0]) | 4e-4 / 1e-3 | 2e-3 / 0.011 | 4e-4 / 2e-3 | 8e-3 / **0.66** | 4e-6 |
| exp, mixed sign ([−4, 2]) | 1e-3 / 0.056 | 3e-3 / 0.011 | 1e-3 / 0.069 | 0.11 / **7.4 ⚠** | 5e-6 |
| log ([0.1, 50]) | 0.020 / 0.033 | 0.020 / 0.033 | 0.020 / 0.032 | 1e-5 / **2.3 ⚠** | 4e-6 |
| reciprocal ([0.1, 50]) | 8e-5 / 6e-4 | 9e-5 / 6e-4 | 9e-5 / 6e-4 | 8e-6 / **2.4e4 ⚠** | 3e-6 |
| inv_sqrt ([0.05, 50]) | 2e-5 / 0.24 | 2e-5 / 0.24 | 2e-5 / 0.24 | **0.76 / 145 ⚠** | 4e-6 |
| sigmoid ([−8, 8]) | 2e-4 / 2e-3 | 1e-3 / 9e-3 | 0.019 / 0.50 | 5e-5 / 4e-3 | 4e-6 |
| tanh ([−6, 6]) | 1e-4 / 4e-3 | 1e-3 / 0.018 | 0.035 / 0.093 | 2e-5 / 9e-3 | 3e-6 |
| erf ([−4, 4]) | **2e3 / 4e6 ⚠** | 2e-3 / 0.015 | 0.027 / 0.071 | 5e-5 / 0.011 | 3e-6 |
| GeLU ([−6, 6]) | **7e3 / 6e7 ⚠** | 4e-4 / 4e-3 | 4e-4 / 4e-3 | 2e-3 / **6.0 ⚠** | 3e-6 |
| SiLU ([−8, 8]) | 6e-4 / 2e-3 | 5e-3 / 0.015 | 6e-4 / 5e-3 | 3e-3 / 0.30 | 4e-6 |
| softmax (N(0, 2) logits) | 1e-4 / 4e-3 | 6e-4 / 6e-3 | 3e-4 / 0.052 | 2e-3 / 0.16 | 4e-6 |
| LayerNorm (N(0.5, 2) inputs) | 9e-6 / 9e-5 | 9e-6 / 8e-5 | 1e-5 / 8e-5 | **0.55 / 19 ⚠** | 4e-6 |

**Known limits:**
- **`curl` (Curl defaults, identical to the Curl reference):**
  - `inv_sqrt` only covers inputs in [0, 2) (`lut_max_bits: 1`), so **LayerNorm is wrong whenever a row's variance is 2 or more**.
  - GeLU only covers |x| < 4 (`lut_max_bits: 2`).
  - `exp` assumes non-positive inputs (`all_neg: True`), so positive inputs return about 0. It is coarse near 0 (error up to about 0.66).
  - Reciprocal and log are inaccurate for inputs near 0.1.
  - About 3–4% of softmax calls return garbage outputs (also seen in the Curl reference).
  - Widen a function's range by raising its `lut_max_bits` (and `bior_size_bits` for resolution) via `prix.cfg.temp_override`.
- **`crypten`:** erf and GeLU use an 8-term Taylor series that diverges for |x| above about 3, the same as in CrypTen.
- **`shaft`:** the Fourier series are periodic, so sigmoid and tanh ring near the edge of the period (|x| close to 8 with the defaults).
- **`poly`:** accurate within about 0.02 on the fitted intervals. `exp` is fitted on [−5, 2.5] and returns 0 below −5.
- **`fission`:** exact to float32 precision. The evaluator sees the permuted plaintext values.

Regenerate this table with [`benchmarks/accuracy_sweep.py`](../benchmarks/accuracy_sweep.py) after changing a configuration (raw results: [`benchmarks/results/accuracy_sweep.json`](../benchmarks/results/accuracy_sweep.json)).

