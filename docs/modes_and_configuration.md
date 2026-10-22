# MPC Approximation Modes & Configuration Reference

PriX supports five nonlinear approximation paradigms out of the box, selectable via `mode="lut" | "poly" | "shaft" | "crypten" | "fission"` in `prix.run`, `@prix.mpc.run_multiprocess`, `prix.init`, or `prix.set_mode`.

---

## 1. Overview of the Five Modes

| Mode Name | Config File | Truncation | Nonlinear Approximations | Reference |
|---|---|---|---|---|
| **`"lut"`** *(default)* | [`configs/default.yaml`](../configs/default.yaml) | `[EGK+20]` probabilistic (`egk`) | Biorthogonal-2.2 & Haar Wavelet LUTs for `exp`, `log`, `reciprocal`, `sqrt`, `inv_sqrt`, `sigmoid`, `tanh`, `erf`, `gelu`, `silu` | [**Curl (CAMLIS 2024)**](https://eprint.iacr.org/2024/1127) |
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

@prix.mpc.run_multiprocess(world_size=2, mode="shaft")
def run_shaft():
    x = prix.array([-1.5, 0.0, 1.5])
    return x.gelu().reconstruct()
```

Or switch dynamically inside a process using `prix.set_mode`:
```python
prix.set_mode("lut")      # Curl Wavelet LUTs
prix.set_mode("poly")     # PUMA / Nimbus piecewise low-degree polynomials
prix.set_mode("shaft")    # SHAFT Fourier + ODE
prix.set_mode("crypten")  # CrypTen classic approximations
prix.set_mode("fission")  # Fission evaluator
```

---

## 3. Mixing & Customizing Methods via `prix.cfg`

Because `prix.cfg` deep-merges every preset onto `default.yaml`, you can freely mix methods from different papers (for example, combining **SHAFT's ODE Softmax** and **1-round Fourier GeLU** with **Curl's Wavelet LUT LayerNorm/InvSqrt** and **EGK truncation**):

```python
import prix

@prix.mpc.run_multiprocess(world_size=2, mode="lut")
def hybrid_curl_shaft():
    # Base mode is "lut" (Curl Wavelet LUTs + EGK truncation);
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
| `encoder.trunc_method.prod` | `"egk"` \| `"crypten"` \| `"fission"` | Truncation protocol after private multiplications / matmuls. |
| `encoder.trunc_method.lut` | `"egk"` \| `"crypten"` \| `"fission"` | Truncation protocol inside wavelet LUT interpolation. |
| `mpc.provider` | `"TFP"` \| `"TTP"` | Correlated randomness dealer (`TrustedFirstParty` vs. `TrustedThirdParty`). |
| `functions.exp.method` | `"bior"` \| `"haar"` \| `"limit"` \| `"fission"` | Exponential approximation (`iterations`, `bior_size_bits`, `lut_max_bits`). |
| `functions.reciprocal.method` | `"bior"` \| `"haar"` \| `"NR"` \| `"fission"` | Reciprocal $1/x$ (`nr_iters`, `bior_size_bits`, `lut_max_bits`). |
| `functions.inv_sqrt.method` | `"bior"` \| `"haar"` \| `"NR"` \| `"fission"` | Inverse square root $1/\sqrt{x}$ used in `LayerNorm` and `RMSNorm`. |
| `functions.gelu.method` | `"bior-lut-only"` \| `"fourier"` \| `"erf"` \| `"poly"` \| `"fission"` | GeLU activation (`fs_period`, `fs_terms` for SHAFT Fourier; `bior_size_bits` for Curl LUT). |
| `functions.silu.method` | `"bior-lut-only"` \| `"fourier"` \| `"sigmoid"` \| `"fission"` | SiLU / Swish activation (`fs_period`, `fs_terms` for SHAFT Fourier). |
| `functions.sigmoid_tanh.method` | `"bior"` \| `"fourier"` \| `"reciprocal"` \| `"chebyshev"` \| `"fission"` | Sigmoid and Tanh activations. |
| `functions.softmax.method` | `"approximation"` \| `"ode"` \| `"fission"` | Softmax (`"ode"` uses SHAFT Forward-Euler ODE with `ode_iter_num`, `ode_ub`, `ode_lb`). |
| `functions.layernorm.method` | `"approximation"` \| `"fission"` | LayerNorm / RMSNorm normalization method. |
