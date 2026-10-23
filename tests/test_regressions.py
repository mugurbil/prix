#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for bugs found in the second audit."""

import unittest

import numpy as np

import prix
import prix.nn as nn


def _large_return():
    # ~800 KB result: used to deadlock the launcher (result queue read only after join).
    return prix.array(np.ones(200_000, np.float32)).reconstruct()


def _matmul_1d():
    v = np.array([1.0, 2.0, 3.0], np.float32)
    w = np.array([0.5, -1.0, 2.0], np.float32)
    m = (np.arange(6, dtype=np.float32).reshape(3, 2) / 4)
    m2 = (np.arange(6, dtype=np.float32).reshape(2, 3) / 4)
    out = {}
    for name, (a, b) in {"vv": (v, w), "vm": (v, m), "mv": (m2, v)}.items():
        priv = np.asarray((prix.array(a) @ prix.array(b)).reconstruct())
        pub_r = np.asarray((prix.array(a) @ b).reconstruct())   # private @ public
        pub_l = np.asarray((a @ prix.array(b)).reconstruct())   # public @ private
        out[name] = (priv, pub_r, pub_l, a @ b)
    # NumPy array on the left of +, -, * must defer to PriX's reflected operators.
    for name, fn in {"radd": lambda a, b: a + b, "rsub": lambda a, b: a - b,
                     "rmul": lambda a, b: a * b}.items():
        got = np.asarray(fn(v, prix.array(w)).reconstruct())
        out[name] = (got, got, got, fn(v, w))
    return out


def _fission_deferred_precision():
    rs = np.random.RandomState(0)
    x = rs.randn(8, 8).astype(np.float32) * 0.5
    w = rs.randn(8, 8).astype(np.float32) * 0.3
    X, W = prix.array(x), prix.array(w)
    a, b = X.matmul(W), X.matmul(W.t())  # deferred truncation under fission
    ap, bp = x @ w, x @ w.T
    return {
        "a@b.T": (np.asarray(a.matmul(b.t()).reconstruct()), ap @ bp.T),
        "x@a": (np.asarray(X.matmul(a).reconstruct()), x @ ap),
        "a*b": (np.asarray((a * b).reconstruct()), ap * bp),
    }


def _mha_vs_plaintext():
    tokens = (np.random.RandomState(1).randn(1, 8, 32) * 0.5).astype(np.float32)
    mha = nn.MultiheadAttention(32, 4)
    plain = np.asarray(mha(tokens, tokens, tokens))
    x = prix.array(tokens, src=1)
    return np.asarray(mha.encrypt(src=0)(x, x, x).reconstruct()), plain


class TestRegressions(unittest.TestCase):
    def test_large_return_value(self):
        out = prix.run(_large_return, world_size=2, timeout=180)
        self.assertEqual(out.shape, (200_000,))
        np.testing.assert_allclose(out, 1.0, atol=1e-3)

    def test_matmul_with_1d_operands(self):
        for name, (priv, pub_r, pub_l, ref) in prix.run(_matmul_1d, world_size=2, timeout=180).items():
            for got in (priv, pub_r, pub_l):
                self.assertEqual(got.shape, ref.shape, name)
                np.testing.assert_allclose(got, ref, atol=1e-2, err_msg=name)

    def test_fission_mul_with_deferred_truncation(self):
        res = prix.run(_fission_deferred_precision, world_size=2, mode="fission", timeout=180)
        for name, (got, ref) in res.items():
            np.testing.assert_allclose(got, ref, atol=1e-2, err_msg=name)

    def test_run_mode_does_not_change_caller_config(self):
        before = prix.cfg.encoder.trunc_method.prod
        prix.run(_large_return, world_size=2, mode="fission", timeout=180)
        self.assertEqual(prix.cfg.encoder.trunc_method.prod, before)

    def test_multihead_attention_matches_plaintext(self):
        for mode in ("shaft", "fission"):
            got, plain = prix.run(_mha_vs_plaintext, world_size=2, mode=mode, timeout=180)
            self.assertEqual(got.shape, plain.shape)
            np.testing.assert_allclose(got, plain, atol=2e-2, err_msg=mode)

    def test_nn_layers_accept_plaintext(self):
        x = np.linspace(-2, 2, 12, dtype=np.float32).reshape(3, 4)
        for layer in (nn.ReLU(), nn.GELU(), nn.SiLU(), nn.Tanh(), nn.Sigmoid(), nn.Softmax(-1),
                      nn.LogSoftmax(-1), nn.Hardtanh(), nn.LayerNorm(4)):
            self.assertEqual(np.asarray(layer(x)).shape, x.shape, type(layer).__name__)


if __name__ == "__main__":
    unittest.main()
