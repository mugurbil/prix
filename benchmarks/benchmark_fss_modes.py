#!/usr/bin/env python3
"""
FSS / wavelet-LUT mode benchmark for PriX on CPU or GPU (2 parties, simulated TFP dealer).

Compares, on identical inputs:
  wave_det      mode="wave"  trunc_method.lut="wave"       2-rd bior(5,3) LUT, (J+j)-level Grotto DPF + j-level DCF
  wave_sigma_tr mode="wave"  trunc_method.lut="sigma"      3-rd bior(5,3) LUT, SIGMA 1-rd TR (j-level DCF) + J-level DPF
  wave_prob     mode="wave"  trunc_method.lut="wave_prob"  2-rd bior(5,3) LUT, local prob. split + J-level DPF
  sigma         mode="sigma"                               SIGMA (PoPETs'24) 0th-order Haar LUT protocols
  curl          mode="lut"                                 Curl 4-rd bior(5,3) LUTs (pre-expanded one-hot + EGK trunc)
  shaft         mode="shaft"                               SHAFT Fourier activations + ODE softmax

For every benchmark we report:
  rounds              online communication rounds
  online_bytes_1p     online bytes sent by ONE party
  key_bytes_1p        preprocessing (dealer key) bytes ONE party would have to receive
                      (PriX's 62-bit PRG seeds; see prix.provider._acct_preproc)
  key_bytes_1p_l128   the same key re-counted with lambda = 128 seed correction words
                      (16 B / DPF level, as in SIGMA / Orca) for a like-for-like comparison
  time_ms / time_s    wall-clock of the online phase on the local device
  mae / max_err       accuracy vs. float64

IMPORTANT: all latencies are *online-phase* numbers with a simulated dealer; no key material
is transferred (both parties expand the same seeds). The key sizes above are what a real
dealer would send. Dealer `Gen` runs inline, so compute is, if anything, over-counted.

Usage:
  python benchmarks/benchmark_fss_modes.py                         # CPU
  python benchmarks/benchmark_fss_modes.py --device cuda           # GPU (single GPU, 2 processes)
  python benchmarks/benchmark_fss_modes.py --modes wave_det,sigma --models bert_base --repeats 3
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import prix  # noqa: E402
import prix.nn as nn  # noqa: E402
from prix.config import cfg  # noqa: E402
from prix.provider import get_preproc_stats, reset_preproc_stats  # noqa: E402

VARIANTS = {
    # key: (mode, trunc_method.lut, trunc_method.prod, title)
    "wave_det": ("wave", "wave", "egk", 'Wave (2-rd Grotto DPF+DCF, trunc_method.lut="wave")'),
    "wave_sigma_tr": ("wave", "sigma", "egk", 'Wave + SIGMA TR (3-rd, trunc_method.lut="sigma")'),
    "wave_prob": ("wave", "wave_prob", "egk", 'Wave-Prob (2-rd, trunc_method.lut="wave_prob")'),
    "sigma": ("sigma", "sigma", "egk", 'SIGMA (mode="sigma")'),
    "curl": ("lut", None, None, 'Curl (mode="lut")'),
    "shaft": ("shaft", None, None, 'SHAFT (mode="shaft")'),
}

MODELS = {
    # name: (d_model, heads, ffn, seq_len, n_layers)
    "bert_base": (768, 12, 3072, 128, 12),
    "bert_large": (1024, 16, 4096, 128, 24),
    "gpt2_64": (768, 12, 3072, 64, 12),
    "gpt2_128": (768, 12, 3072, 128, 12),
}


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = self.attn(x, x, x)
        x = self.ln1(x + h)
        f = self.fc2(self.act(self.fc1(x)))
        x = self.ln2(x + f)
        return x


def _block_until_ready(arr):
    import jax

    jax.block_until_ready(arr.share)


def _timed(fn, repeats, c):
    """Median wall time (s) plus per-run rounds / online bytes / key bytes of `fn()`."""
    out = fn()
    _block_until_ready(out)  # warm-up (JIT compile)
    times, rounds, obytes, kbytes, kbytes_l128 = [], [], [], [], []
    for _ in range(repeats):
        c.barrier()
        c.reset_communication_stats()
        reset_preproc_stats()
        t0 = time.perf_counter()
        out = fn()
        _block_until_ready(out)
        times.append(time.perf_counter() - t0)
        rounds.append(int(c.comm_rounds))
        obytes.append(int(c.comm_bytes))
        ks = get_preproc_stats()
        kbytes.append(ks["bytes"])
        kbytes_l128.append(ks["bytes_l128"])
    return out, {
        "time_s": float(np.median(times)),
        "rounds": int(np.median(rounds)),
        "online_bytes_1p": float(np.median(obytes)),
        "key_bytes_1p": float(np.median(kbytes)),
        "key_bytes_1p_l128": float(np.median(kbytes_l128)),
    }


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(got - ref))), "max_err": float(np.max(np.abs(got - ref)))}


def run_variant(key, n_prim, repeats, models, skip_primitives, skip_sublayers):
    mode, lut_trunc, prod_trunc, title = VARIANTS[key]
    prix.set_mode(mode)
    if lut_trunc is not None:
        cfg.encoder.trunc_method.lut = lut_trunc
    if prod_trunc is not None:
        cfg.encoder.trunc_method.prod = prod_trunc
    c = prix.comm.get()
    c.set_verbosity(True)
    rng = np.random.RandomState(42)
    res = {"title": title, "mode": mode, "trunc_method": dict(cfg.encoder.trunc_method), "primitives": {}, "sublayers": {}, "models": {}}

    if not skip_primitives:
        prim_inputs = {
            "gelu": (np.linspace(-3.8, 3.8, n_prim), lambda x: 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))),
            "silu": (np.linspace(-7.5, 7.5, n_prim), lambda x: x / (1.0 + np.exp(-x))),
            "sigmoid": (np.linspace(-7.5, 7.5, n_prim), lambda x: 1.0 / (1.0 + np.exp(-x))),
            "tanh": (np.linspace(-4.5, 4.5, n_prim), np.tanh),
            "erf": (np.linspace(-3.5, 3.5, n_prim), erf),
            "exp": (np.linspace(-15.0, -0.01, n_prim), np.exp),
            "reciprocal": (np.linspace(1.0, 30.0, n_prim), lambda x: 1.0 / x),
            "inv_sqrt": (np.linspace(0.1, 15.0, n_prim), lambda x: 1.0 / np.sqrt(x)),
        }
        for name, (x_np, ref_fn) in prim_inputs.items():
            x_np = x_np.astype(np.float32)
            x = prix.array(x_np, src=0)
            out, st = _timed(lambda: getattr(x, name)(), repeats, c)
            st["time_ms"] = st.pop("time_s") * 1000.0
            st["online_bytes_per_elem"] = st["online_bytes_1p"] / n_prim
            st["key_bytes_per_elem"] = st["key_bytes_1p"] / n_prim
            st["key_bytes_per_elem_l128"] = st["key_bytes_1p_l128"] / n_prim
            st.update(_err(out.reconstruct(), ref_fn(x_np.astype(np.float64))))
            res["primitives"][name] = st

    if not skip_sublayers:
        g_np = rng.uniform(-3.5, 3.5, size=(1, 128, 3072)).astype(np.float32)
        g = prix.array(g_np, src=0)
        out, st = _timed(lambda: g.gelu(), repeats, c)
        st.update(_err(out.reconstruct(), 0.5 * g_np.astype(np.float64) * (1.0 + erf(g_np.astype(np.float64) / np.sqrt(2.0)))))
        res["sublayers"]["gelu_128x3072"] = st

        s_np = rng.uniform(-4.0, 4.0, size=(1, 12, 128, 128)).astype(np.float32)
        s = prix.array(s_np, src=0)
        out, st = _timed(lambda: s.softmax(axis=-1), repeats, c)
        e = np.exp(s_np.astype(np.float64) - s_np.max(axis=-1, keepdims=True))
        st.update(_err(out.reconstruct(), e / e.sum(axis=-1, keepdims=True)))
        res["sublayers"]["softmax_12x128x128"] = st

        l_np = rng.normal(0.0, 1.0, size=(1, 128, 768)).astype(np.float32)
        ln = nn.LayerNorm(768)
        ln.encrypt()
        l = prix.array(l_np, src=0)
        out, st = _timed(lambda: ln(l), repeats, c)
        mu = l_np.astype(np.float64).mean(-1, keepdims=True)
        var = l_np.astype(np.float64).var(-1, keepdims=True)
        st.update(_err(out.reconstruct(), (l_np - mu) / np.sqrt(var + 1e-5)))
        res["sublayers"]["layernorm_128x768"] = st

    for mname in models:
        d, h, ffn, L, n_layers = MODELS[mname]
        blk = TransformerBlock(d, h, ffn)
        blk.encrypt()
        x_np = rng.normal(0.0, 0.5, size=(1, L, d)).astype(np.float32)
        x = prix.array(x_np, src=0)
        out, st = _timed(lambda: blk(x), max(1, repeats // 2), c)
        st.update({
            "n_layers": n_layers,
            "block_rounds": st["rounds"],
            "total_rounds": st["rounds"] * n_layers,
            "total_online_gb_1p": st["online_bytes_1p"] * n_layers / 1e9,
            "total_online_gb_2p": 2 * st["online_bytes_1p"] * n_layers / 1e9,
            "total_key_gb_1p": st["key_bytes_1p"] * n_layers / 1e9,
            "total_key_gb_1p_l128": st["key_bytes_1p_l128"] * n_layers / 1e9,
            "block_time_s": st["time_s"],
            "total_time_s": st["time_s"] * n_layers,
        })
        res["models"][mname] = st

    if c.get_rank() == 0:
        print(f"=== {title} ===", flush=True)
        for name, st in res["primitives"].items():
            print(f"  {name:10s}: {st['rounds']:2d} rds | {st['online_bytes_per_elem']:6.1f} B | key {st['key_bytes_per_elem']:7.1f} B ({st['key_bytes_per_elem_l128']:7.1f} B @l128) | {st['time_ms']:8.2f} ms | MAE {st['mae']:.2e}", flush=True)
        for name, st in res["sublayers"].items():
            print(f"  {name:18s}: {st['rounds']:3d} rds | {st['online_bytes_1p']/1e6:7.2f} MB | key {st['key_bytes_1p']/1e9:6.3f} GB ({st['key_bytes_1p_l128']/1e9:6.3f} GB @l128) | {st['time_s']:7.3f} s | MAE {st['mae']:.2e}", flush=True)
        for name, st in res["models"].items():
            print(f"  {name:10s}: {st['total_rounds']:5d} rds | online {st['total_online_gb_1p']:6.3f} GB 1P | key {st['total_key_gb_1p']:7.3f} GB 1P ({st['total_key_gb_1p_l128']:7.3f} GB @l128) | {st['total_time_s']:7.2f} s ({st['block_time_s']:.2f} s/blk)", flush=True)
    return res


def _party_main(args):
    import jax

    out = {}
    for key in args.variants:
        out[key] = run_variant(key, args.n_prim, args.repeats, args.models, args.skip_primitives, args.skip_sublayers)
    out["_env"] = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax": jax.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        out["_env"]["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    p.add_argument("--modes", default=",".join(VARIANTS), help=f"comma-separated subset of {list(VARIANTS)}")
    p.add_argument("--models", default="bert_base,bert_large,gpt2_64", help=f"comma-separated subset of {list(MODELS)} (or '')")
    p.add_argument("--n-prim", type=int, default=1000, help="elements per primitive benchmark")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-primitives", action="store_true")
    p.add_argument("--skip-sublayers", action="store_true")
    p.add_argument("--out", default=None, help="JSON output path (default: benchmarks/results/fss_modes_<device>.json)")
    args = p.parse_args()
    args.variants = [v for v in args.modes.split(",") if v]
    args.models = [m for m in args.models.split(",") if m]
    for v in args.variants:
        assert v in VARIANTS, f"unknown variant {v!r}"
    for m in args.models:
        assert m in MODELS, f"unknown model {m!r}"

    results = prix.run(_party_main, args, world_size=2, device=args.device)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"fss_modes_{args.device.replace(':', '')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
