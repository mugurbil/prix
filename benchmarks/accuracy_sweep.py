"""
Empirical accuracy of every PriX nonlinearity in every mode (as shipped), vs exact float64.
Each function is evaluated REPS times on a fixed grid to catch intermittent failures.

Usage (from the repository root):
    JAX_PLATFORMS=cpu python benchmarks/accuracy_sweep.py
Results are written to benchmarks/results/accuracy_sweep.json (override with OUT=...).
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPS = int(os.environ.get("REPS", "10"))
rng = np.random.default_rng(0)

GRIDS = {
    "exp(neg)": np.linspace(-8, 0, 257),
    "exp(mixed)": np.linspace(-4, 2, 241),
    "log": np.linspace(0.1, 50, 250),
    "reciprocal": np.linspace(0.1, 50, 250),
    "inv_sqrt": np.linspace(0.05, 50, 250),
    "sigmoid": np.linspace(-8, 8, 257),
    "tanh": np.linspace(-6, 6, 241),
    "erf": np.linspace(-4, 4, 257),
    "gelu": np.linspace(-6, 6, 241),
    "silu": np.linspace(-8, 8, 257),
}
SOFTMAX_IN = rng.normal(0, 2.0, size=(8, 32))
LN_IN = rng.normal(0.5, 2.0, size=(8, 64))


def _erf(x):
    return np.vectorize(math.erf)(x)


REF = {
    "exp(neg)": np.exp, "exp(mixed)": np.exp, "log": np.log, "reciprocal": lambda x: 1 / x,
    "inv_sqrt": lambda x: 1 / np.sqrt(x), "sigmoid": lambda x: 1 / (1 + np.exp(-x)), "tanh": np.tanh,
    "erf": _erf, "gelu": lambda x: 0.5 * x * (1 + _erf(x / math.sqrt(2))), "silu": lambda x: x / (1 + np.exp(-x)),
}


def worker(mode):
    import prix

    out = {}
    calls = {
        "exp(neg)": lambda a: a.exp(), "exp(mixed)": lambda a: a.exp(), "log": lambda a: a.log(),
        "reciprocal": lambda a: a.reciprocal(), "inv_sqrt": lambda a: a.inv_sqrt(), "sigmoid": lambda a: a.sigmoid(),
        "tanh": lambda a: a.tanh(), "erf": lambda a: a.erf(), "gelu": lambda a: a.gelu(), "silu": lambda a: a.silu(),
    }
    for name, fn in calls.items():
        x = prix.array(GRIDS[name].astype(np.float32))
        try:
            out[name] = [np.asarray(fn(x).reconstruct(), dtype=np.float64).tolist() for _ in range(REPS)]
        except Exception as e:  # noqa: BLE001
            out[name] = f"ERROR: {type(e).__name__}: {e}"
    s = prix.array(SOFTMAX_IN.astype(np.float32))
    out["softmax"] = [np.asarray(s.softmax(dim=-1).reconstruct(), dtype=np.float64).tolist() for _ in range(REPS)]
    ln = prix.array(LN_IN.astype(np.float32))
    out["layernorm"] = [np.asarray(ln.layernorm().reconstruct(), dtype=np.float64).tolist() for _ in range(REPS)]
    # Write to a file instead of returning: prix.run currently hangs on return values > ~64 KB.
    if prix.comm.get().get_rank() == 0:
        json.dump(out, open(os.path.join(tempfile.gettempdir(), f"_acc_{mode}.json"), "w"))


def summarize(name, runs, ref):
    if isinstance(runs, str):
        return {"error": runs}
    arr = np.array(runs)
    err = np.abs(arr - ref)
    per_run_max = err.reshape(len(runs), -1).max(axis=1)
    return {
        "median_abs": float(np.median(err)),
        "p99_abs": float(np.quantile(err, 0.99)),
        "max_abs_typical_run": float(np.median(per_run_max)),
        "worst_run_max_abs": float(per_run_max.max()),
        "runs_with_err_gt_1": int((per_run_max > 1.0).sum()),
        "reps": len(runs),
    }


if __name__ == "__main__":
    import prix

    modes = os.environ.get("MODES", "crypten,poly,shaft,lut,fission").split(",")
    sm_ref = np.exp(SOFTMAX_IN - SOFTMAX_IN.max(-1, keepdims=True))
    sm_ref /= sm_ref.sum(-1, keepdims=True)
    ln_ref = (LN_IN - LN_IN.mean(-1, keepdims=True)) / np.sqrt(LN_IN.var(-1, keepdims=True) + 1e-5)
    results = {}
    for mode in modes:
        prix.run(worker, mode, world_size=2, mode=mode, start_method="fork", timeout=1800)
        raw = json.load(open(os.path.join(tempfile.gettempdir(), f"_acc_{mode}.json")))
        res = {name: summarize(name, raw[name], REF[name](GRIDS[name])) for name in GRIDS}
        res["softmax"] = summarize("softmax", raw["softmax"], sm_ref)
        res["layernorm"] = summarize("layernorm", raw["layernorm"], ln_ref)
        results[mode] = res
        print(f"done {mode}", flush=True)
    out = os.environ.get("OUT", "benchmarks/results/accuracy_sweep.json")
    json.dump(results, open(out, "w"), indent=1)
    names = list(GRIDS) + ["softmax", "layernorm"]
    print("\n| Function | " + " | ".join(modes) + " |")
    print("|---|" + "---:|" * len(modes))
    for n in names:
        cells = []
        for m in modes:
            r = results[m][n]
            if "error" in r:
                cells.append("error")
            else:
                bad = f" ⚠{r['runs_with_err_gt_1']}/{r['reps']}" if r["runs_with_err_gt_1"] else ""
                cells.append(f"{r['median_abs']:.3g} / {r['max_abs_typical_run']:.3g}{bad}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\ncells: median abs error / typical per-run max abs error; ⚠k/n = runs with max error > 1")
