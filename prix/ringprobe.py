#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Ring-width probe (study instrumentation, off by default).

PriX computes over Z_{2^64}. To decide whether a smaller ring Z_{2^k} (e.g. k = 32) would be
correct for a workload, it suffices to know the magnitude of every value at the points where a
k-bit ring would first break:

  - ``trunc_egk``   inputs to [EGK+20] TruncPr, which needs |x| < 2^(l-1) with l <= k - 2;
  - ``trunc_local`` inputs to local (SecureML-style) share division, which fails with
                    probability ~ |x| / 2^(k-1) per element;
  - ``cmp``         inputs to secure comparisons (DCF/DPF or carry-lookahead), which need
                    |x| < 2^(k-2) for the sign to be meaningful;
  - ``sigma_tr``    inputs to SIGMA's faithful Truncate-Reduce, which need |x| < 2^(k-1).

With ``PRIX_RING_PROBE=1`` every such value is reconstructed (an extra opening, so timings and
round counts are NOT meaningful in this mode) and summarised per site. Nothing else changes:
results are bit-identical to an unprobed run.
"""

import math
import os

import numpy as np

ENABLED = os.environ.get("PRIX_RING_PROBE", "0") == "1"
RING_BITS = (32,)  # candidate ring widths to evaluate

_STATS = {}


def enabled():
    return ENABLED


def reset():
    _STATS.clear()


def stats():
    """Per-site summary: element count, max |x| (log2), and per-k violation counts."""
    out = {}
    for tag, s in _STATS.items():
        d = {"n": int(s["n"]), "calls": int(s["calls"]),
             "max_abs_log2": (math.log2(s["max"]) if s["max"] > 0 else None)}
        for k in RING_BITS:
            d[f"viol_k{k}"] = int(s[f"viol_k{k}"])
            if tag == "trunc_local":
                d[f"expected_fail_k{k}"] = float(s[f"efail_k{k}"])
        d["log2_hist"] = {str(b): int(c) for b, c in sorted(s["hist"].items())}
        out[tag] = d
    return out


def _threshold(tag, k):
    """|x| at or above which a k-bit ring is incorrect for this site."""
    if tag == "trunc_egk":
        return 2.0 ** (k - 3)  # l = k - 2, TruncPr requires |x| < 2^(l-1)
    if tag == "cmp":
        return 2.0 ** (k - 2)
    return 2.0 ** (k - 1)  # trunc_local (deterministic wrap), sigma_tr


def record(tag, share):
    """Reconstructs `share` (int64 additive shares) and accumulates magnitude statistics."""
    if not ENABLED:
        return
    from prix import communicator as comm

    c = comm.get()
    v = np.asarray(c.all_reduce(share, op="sum"), dtype=np.int64).ravel()
    if c.get_rank() != 0 or v.size == 0:
        return
    a = np.abs(v.astype(np.float64))
    s = _STATS.setdefault(tag, {"n": 0, "calls": 0, "max": 0.0, "hist": {},
                                **{f"viol_k{k}": 0 for k in RING_BITS},
                                **{f"efail_k{k}": 0.0 for k in RING_BITS}})
    s["n"] += a.size
    s["calls"] += 1
    s["max"] = max(s["max"], float(a.max()))
    for k in RING_BITS:
        s[f"viol_k{k}"] += int(np.count_nonzero(a >= _threshold(tag, k)))
        if tag == "trunc_local":
            s[f"efail_k{k}"] += float(np.minimum(1.0, a / 2.0 ** (k - 1)).sum())
    b, cnt = np.unique(np.floor(np.log2(np.maximum(a, 1.0))).astype(np.int64), return_counts=True)
    for bi, ci in zip(b.tolist(), cnt.tolist()):
        s["hist"][bi] = s["hist"].get(bi, 0) + ci
