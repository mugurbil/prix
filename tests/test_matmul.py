#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Bit-exactness tests for the limb-split float64 ring matmul, including K-chunking.

The checks run in a *spawned* child process: prix's multiprocess launcher forks, and
forking a process whose XLA runtime has already executed work can deadlock the
children. Keeping XLA out of the pytest process lets these tests run in the same
session as the multiprocess tests in test_prix.py.
"""

import multiprocessing
import unittest


def _ref_matmul(a, b):
    """Reference int64 matmul with exact wrap-around mod 2^64 (Python ints)."""
    import numpy as np

    out = np.matmul(np.asarray(a).astype(object), np.asarray(b).astype(object))
    wrap = np.vectorize(lambda v: ((int(v) + 2**63) % 2**64) - 2**63, otypes=[object])
    return wrap(out).astype(np.int64)


def _run_checks():
    """Runs all matmul checks; returns a list of failure messages (empty on success)."""
    from unittest import mock

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    import prix.provider as provider

    rng = np.random.default_rng(0)
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    rand = lambda shape: rng.integers(lo, hi, size=shape, dtype=np.int64)
    failures = []

    def check(sa, sb, label):
        a, b = rand(sa), rand(sb)
        got = np.asarray(provider._int64_matmul(jnp.asarray(a), jnp.asarray(b)))
        if not np.array_equal(got, _ref_matmul(a, b)):
            failures.append(f"{label}: mismatch for shapes {sa} @ {sb}")

    # K <= 512 uses 3x22-bit limbs, K > 512 uses 4x16-bit limbs; full-range ring values.
    for k in (1, 128, 129, 512, 513, 1024):
        check((3, k), (k, 4), "exact")

    # Shrink the limits so small matrices exercise the chunking path. disable_jit makes the
    # patched module constants visible on every call (no stale traced constants).
    with mock.patch.object(provider, "_MATMUL_MAX_K_22", 4), \
            mock.patch.object(provider, "_MATMUL_MAX_K_16", 16), jax.disable_jit():
        for sa, sb in [
            ((3, 64), (64, 5)),        # 4 full 16-bit chunks
            ((3, 70), (70, 5)),        # remainder 6 -> 16-bit limbs
            ((3, 66), (66, 5)),        # remainder 2 -> 22-bit limbs
            ((2, 3, 40), (40, 5)),     # batched lhs
            ((2, 3, 40), (2, 40, 5)),  # batched lhs and rhs
            ((40,), (40, 5)),          # 1-D lhs
            ((3, 40), (40,)),          # 1-D rhs
        ]:
            check(sa, sb, "chunked")

        # Confirm the chunked path is really taken: K=70 -> chunks of 16,16,16,16,6.
        with mock.patch.object(
            provider, "_int64_matmul_exact", wraps=provider._int64_matmul_exact
        ) as spy:
            check((3, 70), (70, 5), "chunk-split")
            ks = [c.args[0].shape[-1] for c in spy.call_args_list]
            if ks != [16, 16, 16, 16, 6]:
                failures.append(f"chunk-split: expected chunks [16,16,16,16,6], got {ks}")
    return failures


class TestInt64Matmul(unittest.TestCase):
    def test_limb_matmul_exact_and_chunked(self):
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            failures = pool.apply(_run_checks)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
