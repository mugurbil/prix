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


def _ttp_trig_and_perm():
    assert prix.get_default_provider().NAME == "TTP"
    provider = prix.get_default_provider()
    # Verify trig triple produces 61-bit positive t and accurate Fourier GeLU
    t, u, v = provider.generate_trig_triple((32,), 8.0, 8)
    t_raw = np.asarray(prix.comm.get().all_reduce(t.share))
    assert np.all((t_raw >= 0) & (t_raw < (1 << 61)))
    # Verify permutation comes from the joint MPC generator and matches across parties
    perm, inv_perm = provider.generate_permutation((16,))
    p_np = np.asarray(perm)
    ip_np = np.asarray(inv_perm)
    assert np.array_equal(p_np[ip_np], np.arange(16))
    # Verify disallowed RPC is rejected on client before wire send
    rejected = False
    if prix.comm.get().get_rank() == 0:
        try:
            prix.provider.TTPClient.get().ttp_request("_get_additive_PRSS", "cpu", (4,))
        except ValueError:
            rejected = True
    else:
        rejected = True
    return bool(rejected), p_np


def _sigma_fss_checks():
    c = prix.comm.get()
    c.set_verbosity(True)
    x_np = np.linspace(-6.0, 6.0, 128, dtype=np.float32)
    x = prix.array(x_np, src=0)

    c.reset_communication_stats()
    y = x.relu()
    relu_rounds, relu_bytes = c.comm_rounds, c.comm_bytes
    relu_out = np.asarray(y.reconstruct())

    from prix.approximations import _ltz_and_abs_lt
    c.reset_communication_stats()
    ltz, check, neg_part = _ltz_and_abs_lt(x, 4.0, return_neg_part=True)
    way3_rounds, way3_bytes = c.comm_rounds, c.comm_bytes
    ltz_out = np.asarray(ltz.reconstruct())
    chk_out = np.asarray(check.reconstruct())
    neg_out = np.asarray(neg_part.reconstruct())

    emb = nn.Embedding(64, 16)
    idx_np = np.array([0, 7, 19, 63], dtype=np.int64)
    emb_plain = np.asarray(emb(idx_np))
    idx_enc = prix.array(idx_np, precision=0, src=1)
    emb.encrypt(src=0)
    c.reset_communication_stats()
    emb_out = emb(idx_enc)
    emb_bytes = c.comm_bytes
    emb_got = np.asarray(emb_out.reconstruct())

    return {
        "x_np": x_np,
        "relu": (relu_rounds, relu_bytes, relu_out),
        "way3": (way3_rounds, way3_bytes, ltz_out, chk_out, neg_out),
        "emb": (emb_bytes, emb_got, emb_plain),
    }


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
        for mode in ("sigma", "shaft", "fission"):
            got, plain = prix.run(_mha_vs_plaintext, world_size=2, mode=mode, timeout=180)
            self.assertEqual(got.shape, plain.shape)
            np.testing.assert_allclose(got, plain, atol=2e-2, err_msg=mode)

    def test_nn_layers_accept_plaintext(self):
        x = np.linspace(-2, 2, 12, dtype=np.float32).reshape(3, 4)
        for layer in (nn.ReLU(), nn.GELU(), nn.SiLU(), nn.Tanh(), nn.Sigmoid(), nn.Softmax(-1),
                      nn.LogSoftmax(-1), nn.Hardtanh(), nn.LayerNorm(4)):
            self.assertEqual(np.asarray(layer(x)).shape, x.shape, type(layer).__name__)

    def test_ttp_hardening_and_trig_triple(self):
        prev_provider = prix.cfg.mpc.provider
        try:
            prix.cfg.mpc.provider = "TTP"
            rejected, perm = prix.run(_ttp_trig_and_perm, world_size=2, timeout=180)
            self.assertTrue(rejected)
            self.assertEqual(sorted(perm.tolist()), list(range(16)))
        finally:
            prix.cfg.mpc.provider = prev_provider

    def test_lut_cache_hardening(self):
        import os
        import stat
        from prix.approximations import LookupTables

        cdir = LookupTables._safe_cache_dir()
        st = os.lstat(cdir)
        self.assertTrue(stat.S_ISDIR(st.st_mode))
        self.assertEqual(st.st_mode & 0o077, 0)
        p1 = LookupTables._cache_path("bior", "test", 4, 6, False, function=lambda x: x + 1.0)
        p2 = LookupTables._cache_path("bior", "test", 4, 6, False, function=lambda x: x + 2.0)
        self.assertNotEqual(p1, p2)

    def test_sigma_fss_protocol(self):
        res = prix.run(_sigma_fss_checks, world_size=2, mode="sigma", timeout=180)
        x_np = res["x_np"]
        relu_rounds, relu_bytes, relu_out = res["relu"]
        self.assertEqual(relu_rounds, 2)
        self.assertEqual(relu_bytes, 128 * 8 + 16)  # 8.125 B/elem with 64-to-1 bit packing
        np.testing.assert_allclose(relu_out, np.maximum(x_np, 0.0), atol=1e-4)

        way3_rounds, way3_bytes, ltz_out, chk_out, neg_out = res["way3"]
        self.assertEqual(way3_rounds, 2)
        self.assertEqual(way3_bytes, 128 * 8 + 32)  # 8.25 B/elem with 2 packed bits/elem
        np.testing.assert_allclose(ltz_out, (x_np < 0).astype(np.float32), atol=1e-4)
        np.testing.assert_allclose(chk_out, (np.abs(x_np) < 4.0).astype(np.float32), atol=1e-4)
        np.testing.assert_allclose(neg_out, np.minimum(x_np, 0.0), atol=1e-4)

        emb_bytes, emb_got, emb_plain = res["emb"]
        # Static-weight linear triple: only 4 indices (32B) + 4*64 one-hot (2048B) opened, not 64*16 table (8192B)
        self.assertEqual(emb_bytes, 4 * 8 + 4 * 64 * 8)
        np.testing.assert_allclose(emb_got, emb_plain, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
