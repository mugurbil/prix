"""
Ring-width study (plan WP-0): could PriX run over Z_{2^32} instead of Z_{2^64}?

For every mode and fixed-point precision f, runs the paper's sublayers and one BERT-base block
(L = 128) with the ring probe enabled (prix/ringprobe.py). The probe reconstructs every value at
the sites where a k-bit ring would first break (TruncPr inputs, local truncation inputs,
comparison inputs, LUT/Truncate-Reduce inputs) and counts violations for k = 32. Accuracy is
measured in the 64-bit ring at precision f, which is exactly what a 32-bit ring would compute
whenever the violation count is zero.

Timings and rounds are NOT meaningful here (the probe adds openings).

Usage (from the repository root):
    JAX_PLATFORMS=cpu python benchmarks/ring_sweep.py                       # all modes, f in 16,12,10,8
    JAX_PLATFORMS=cpu python benchmarks/ring_sweep.py --modes default,shaft --precisions 16,12
Writes benchmarks/results/ring_sweep.json (override with --out).
"""
import os

os.environ["PRIX_RING_PROBE"] = "1"  # must be set before prix is imported (here and in the parties)

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import erf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import prix  # noqa: E402
import prix.nn as nn  # noqa: E402
from prix import ringprobe  # noqa: E402
from prix.config import cfg  # noqa: E402
from benchmark_fss_modes import EXTRA_OVERRIDES, TransformerBlock  # noqa: E402

# variant -> (mode, config overrides). Fission is excluded: its evaluator works at 2f bits in the clear.
VARIANTS = {
    "default": ("default", {}),
    "wave": ("wave", {}),
    "sigma_ext": ("sigma", EXTRA_OVERRIDES["sigma_ext"]),
    "curl": ("curl", {}),
    "shaft": ("shaft", {}),
    "poly": ("poly", {}),
    "crypten": ("crypten", {}),
}


def _err(got, ref):
    got = np.asarray(got, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(got - ref))), "max_err": float(np.max(np.abs(got - ref)))}


def _measure(name, fn, ref):
    ringprobe.reset()
    try:
        out = fn()
        res = _err(out.reconstruct(), ref)
    except Exception as e:  # noqa: BLE001  (a wrapped value can make a protocol raise)
        res = {"error": f"{type(e).__name__}: {e}"[:300]}
    res["probe"] = ringprobe.stats()
    return name, res


def worker(args):
    variant, f, seq = args
    mode, overrides = VARIANTS[variant]
    prix.set_mode(mode)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.encoder.precision_bits = int(f)
    rng = np.random.RandomState(42)
    out = {}

    g_np = rng.uniform(-3.5, 3.5, size=(1, seq, 3072)).astype(np.float32)
    g64 = g_np.astype(np.float64)
    k, v = _measure("gelu", lambda: prix.array(g_np, src=0).gelu(), 0.5 * g64 * (1.0 + erf(g64 / np.sqrt(2.0))))
    out[k] = v

    s_np = rng.uniform(-4.0, 4.0, size=(1, 12, seq, seq)).astype(np.float32)
    e = np.exp(s_np.astype(np.float64) - s_np.max(axis=-1, keepdims=True))
    k, v = _measure("softmax", lambda: prix.array(s_np, src=0).softmax(axis=-1), e / e.sum(axis=-1, keepdims=True))
    out[k] = v

    l_np = rng.normal(0.0, 1.0, size=(1, seq, 768)).astype(np.float32)
    ln = nn.LayerNorm(768)
    ln.encrypt()
    l64 = l_np.astype(np.float64)
    k, v = _measure("layernorm", lambda: ln(prix.array(l_np, src=0)),
                    (l64 - l64.mean(-1, keepdims=True)) / np.sqrt(l64.var(-1, keepdims=True) + 1e-5))
    out[k] = v

    np.random.seed(123)
    blk = TransformerBlock(768, 12, 3072)
    x_np = rng.normal(0.0, 0.5, size=(1, seq, 768)).astype(np.float32)
    ref = np.asarray(blk(x_np), dtype=np.float64)
    blk.encrypt()
    k, v = _measure("bert_base_block", lambda: blk(prix.array(x_np, src=0)), ref)
    out[k] = v
    return out


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(HERE), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", default=",".join(VARIANTS))
    p.add_argument("--precisions", default="16,12,10,8")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--out", default=os.path.join(HERE, "results", "ring_sweep.json"))
    a = p.parse_args()
    modes = [m for m in a.modes.split(",") if m]
    precs = [int(x) for x in a.precisions.split(",") if x]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for m in modes:
        for f in precs:
            print(f"--- {m} f={f}", flush=True)
            r = prix.run(worker, (m, f, a.seq), world_size=2, start_method="fork", timeout=3600)
            res.setdefault(m, {})[str(f)] = r
            blk = r["bert_base_block"]
            viol = {t: s.get("viol_k32") for t, s in blk["probe"].items()}
            print(f"    block MAE {blk.get('mae', blk.get('error'))}  k32 violations {viol}", flush=True)
            res["_env"] = {"git_commit": _git_commit(), "seq": a.seq, "ring_bits_probed": list(ringprobe.RING_BITS),
                           "host": platform.node(), "jax": __import__("jax").__version__, "python": platform.python_version(),
                           "loadavg_1m": os.getloadavg()[0], "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
