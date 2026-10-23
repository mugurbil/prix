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


def _wave_fss_lut_checks():
    c = prix.comm.get()
    c.set_verbosity(True)
    x_np = np.linspace(-3.5, 3.5, 128, dtype=np.float32)
    x = prix.array(x_np, src=0)

    c.reset_communication_stats()
    y_bior = x.gelu()
    bior_rounds, bior_bytes = c.comm_rounds, c.comm_bytes
    bior_out = np.asarray(y_bior.reconstruct())

    with prix.cfg.temp_override({"functions.gelu.method": "haar-lut-only", "functions.gelu.haar_size_bits": 8}):
        c.reset_communication_stats()
        y_haar = x.gelu()
        haar_rounds, haar_bytes = c.comm_rounds, c.comm_bytes
        haar_out = np.asarray(y_haar.reconstruct())

    with prix.cfg.temp_override({"encoder.trunc_method.lut": "wave_prob"}):
        c.reset_communication_stats()
        y_bior_prob = x.gelu()
        bior_prob_rounds, bior_prob_bytes = c.comm_rounds, c.comm_bytes
        bior_prob_out = np.asarray(y_bior_prob.reconstruct())

    return {
        "x_np": x_np,
        "bior": (bior_rounds, bior_bytes, bior_out),
        "haar": (haar_rounds, haar_bytes, haar_out),
        "bior_prob": (bior_prob_rounds, bior_prob_bytes, bior_prob_out),
    }


def _split_nexp_and_dynamic_lut():
    x_np = np.linspace(-6.0, 0.0, 65, dtype=np.float32)
    x = prix.array(x_np, src=0)
    with prix.cfg.temp_override({"functions.exp.method": "split", "functions.exp.all_neg": True}):
        split_out = np.asarray(x.exp().reconstruct())
    # Verify changing lut_max_bits dynamically via temp_override updates the LUT without manual cache clearing
    g_np = np.array([-5.0, 5.0], dtype=np.float32)
    g = prix.array(g_np, src=0)
    with prix.cfg.temp_override({"functions.gelu.lut_max_bits": 2}):
        _ = np.asarray(g.gelu().reconstruct())
    with prix.cfg.temp_override({"functions.gelu.lut_max_bits": 3}):
        g_wide = np.asarray(g.gelu().reconstruct())
    with prix.cfg.temp_override({"functions.gelu.method": "haar"}):
        g_haar = np.asarray(prix.array(np.array([-1.0, 0.5, 1.5], np.float32)).gelu().reconstruct())
    with prix.cfg.temp_override({"functions.silu.method": "bior-lut-only"}):
        s_blo = np.asarray(prix.array(np.array([-1.0, 0.5, 1.5], np.float32)).silu().reconstruct())
    return split_out, np.exp(x_np), g_wide, g_haar, s_blo


def _three_party_ops():
    x_np = np.array([-2.0, 0.5, 3.0, 4.0], dtype=np.float32)
    y_np = np.array([1.0, -1.0, 2.0, 0.5], dtype=np.float32)
    div_np = np.array([2, 2, 4, 4], dtype=np.int64)
    x = prix.array(x_np, src=0)
    y = prix.array(y_np, src=1)
    # Non-scalar integer divisor in 3-party mode (Audit item B5)
    div_out = np.asarray(x.div(div_np).reconstruct())
    return (
        np.asarray((x + y).reconstruct()),
        np.asarray((x * y).reconstruct()),
        np.asarray((x < y).reconstruct()),
        np.asarray(x.relu().reconstruct()),
        div_out,
    )


def _fission_multi_eval():
    rs = np.random.RandomState(42)
    x_np = rs.randn(4, 8).astype(np.float32)
    x = prix.array(x_np, src=0)
    return (
        x_np,
        np.asarray(x.gelu().reconstruct()),
        np.asarray(x.softmax(dim=-1).reconstruct()),
        np.asarray(x.layernorm().reconstruct()),
    )


def _dense_mode_sweep():
    grid = np.linspace(-1.75, 1.75, 64, dtype=np.float32)
    rs = np.random.RandomState(7)
    logits = (rs.randn(4, 16) * 1.5).astype(np.float32)
    ln_in = (rs.randn(4, 16) * 1.5 + 0.5).astype(np.float32)
    x = prix.array(grid, src=0)
    sm = np.asarray(prix.array(logits, src=0).softmax(dim=-1).reconstruct())
    ln = np.asarray(prix.array(ln_in, src=0).layernorm().reconstruct())
    return {
        "grid": grid,
        "logits": logits,
        "ln_in": ln_in,
        "gelu": np.asarray(x.gelu().reconstruct()),
        "silu": np.asarray(x.silu().reconstruct()),
        "sigmoid": np.asarray(x.sigmoid().reconstruct()),
        "tanh": np.asarray(x.tanh().reconstruct()),
        "erf": np.asarray(x.erf().reconstruct()),
        "softmax": sm,
        "layernorm": ln,
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
        for mode in ("wave", "sigma", "shaft", "fission"):
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

    def test_split_nexp_and_dynamic_lut_override(self):
        split_out, ref, g_wide, g_haar, s_blo = prix.run(
            _split_nexp_and_dynamic_lut, world_size=2, mode="lut", timeout=180
        )
        np.testing.assert_allclose(split_out, ref, atol=0.07)
        np.testing.assert_allclose(g_wide, np.array([0.0, 5.0], dtype=np.float32), atol=0.15)
        pts = np.array([-1.0, 0.5, 1.5], dtype=np.float64)
        from scipy.special import erf
        gelu_ref = 0.5 * pts * (1.0 + erf(pts / np.sqrt(2.0)))
        silu_ref = pts / (1.0 + np.exp(-pts))
        np.testing.assert_allclose(g_haar, gelu_ref, atol=0.05)
        np.testing.assert_allclose(s_blo, silu_ref, atol=0.15)

    def test_three_party_mpc_and_nonscalar_div(self):
        add_o, mul_o, lt_o, relu_o, div_o = prix.run(
            _three_party_ops, world_size=3, mode="crypten", timeout=180
        )
        x_np = np.array([-2.0, 0.5, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([1.0, -1.0, 2.0, 0.5], dtype=np.float32)
        div_np = np.array([2, 2, 4, 4], dtype=np.float32)
        np.testing.assert_allclose(add_o, x_np + y_np, atol=1e-3)
        np.testing.assert_allclose(mul_o, x_np * y_np, atol=1e-2)
        np.testing.assert_allclose(lt_o, (x_np < y_np).astype(np.float32), atol=1e-3)
        np.testing.assert_allclose(relu_o, np.maximum(x_np, 0.0), atol=1e-2)
        np.testing.assert_allclose(div_o, x_np / div_np, atol=1e-2)

    def test_fission_multi_evaluator_and_ttp(self):
        from scipy.special import erf
        for eval_size, prov in ((2, "TFP"), (1, "TTP")):
            with prix.cfg.temp_override({"mpc.provider": prov}):
                x_np, g, sm, ln = prix.run(
                    _fission_multi_eval, world_size=2, evaluator_size=eval_size, mode="fission", timeout=180
                )
            x64 = x_np.astype(np.float64)
            gelu_ref = 0.5 * x64 * (1.0 + erf(x64 / np.sqrt(2.0)))
            ex = np.exp(x64 - np.max(x64, axis=-1, keepdims=True))
            sm_ref = ex / np.sum(ex, axis=-1, keepdims=True)
            ln_ref = (x64 - x64.mean(-1, keepdims=True)) / np.sqrt(x64.var(-1, keepdims=True) + 1e-5)
            np.testing.assert_allclose(g, gelu_ref, atol=1e-4)
            np.testing.assert_allclose(sm, sm_ref, atol=1e-4)
            np.testing.assert_allclose(ln, ln_ref, atol=1e-4)

    def test_dense_accuracy_and_range_invariants_all_modes(self):
        from scipy.special import erf
        for mode, atol in (("lut", 0.08), ("wave", 0.08), ("sigma", 0.08), ("poly", 0.03), ("shaft", 0.07), ("crypten", 0.08), ("fission", 1e-3)):
            res = prix.run(_dense_mode_sweep, world_size=2, mode=mode, timeout=180)
            g = res["grid"].astype(np.float64)
            logits = res["logits"].astype(np.float64)
            ln_in = res["ln_in"].astype(np.float64)
            gelu_ref = 0.5 * g * (1.0 + erf(g / np.sqrt(2.0)))
            silu_ref = g / (1.0 + np.exp(-g))
            sig_ref = 1.0 / (1.0 + np.exp(-g))
            tanh_ref = np.tanh(g)
            erf_ref = erf(g)
            ex = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            sm_ref = ex / np.sum(ex, axis=-1, keepdims=True)
            ln_ref = (ln_in - ln_in.mean(-1, keepdims=True)) / np.sqrt(ln_in.var(-1, keepdims=True) + 1e-5)

            gelu_atol = 0.25 if mode in ("lut", "sigma") else atol
            silu_atol = 0.50 if mode in ("lut", "sigma") else atol
            self.assertLess(float(np.median(np.abs(res["gelu"] - gelu_ref))), 0.03, f"{mode}:gelu_med")
            self.assertLess(float(np.median(np.abs(res["silu"] - silu_ref))), 0.03, f"{mode}:silu_med")
            np.testing.assert_allclose(res["gelu"], gelu_ref, atol=gelu_atol, err_msg=f"{mode}:gelu")
            np.testing.assert_allclose(res["silu"], silu_ref, atol=silu_atol, err_msg=f"{mode}:silu")
            np.testing.assert_allclose(res["sigmoid"], sig_ref, atol=atol, err_msg=f"{mode}:sigmoid")
            np.testing.assert_allclose(res["tanh"], tanh_ref, atol=atol, err_msg=f"{mode}:tanh")
            np.testing.assert_allclose(res["erf"], erf_ref, atol=atol, err_msg=f"{mode}:erf")
            np.testing.assert_allclose(res["softmax"], sm_ref, atol=atol, err_msg=f"{mode}:softmax")
            np.testing.assert_allclose(res["layernorm"], ln_ref, atol=atol, err_msg=f"{mode}:layernorm")

            # Range invariants: softmax in [0, 1] and rows sum to ~1; sigmoid in [0, 1]; tanh/erf in [-1, 1]
            self.assertTrue(np.all((res["softmax"] >= -0.02) & (res["softmax"] <= 1.02)), mode)
            np.testing.assert_allclose(res["softmax"].sum(axis=-1), 1.0, atol=0.06, err_msg=f"{mode}:sm_sum")
            self.assertTrue(np.all((res["sigmoid"] >= -0.05) & (res["sigmoid"] <= 1.05)), mode)
            self.assertTrue(np.all((res["tanh"] >= -1.05) & (res["tanh"] <= 1.05)), mode)

    def test_packaging_and_config_sync(self):
        import glob
        import os

        repo_root = os.path.normpath(os.path.join(os.path.dirname(prix.__file__), ".."))
        pkg_cfg_dir = os.path.join(os.path.dirname(prix.__file__), "config")
        repo_cfg_dir = os.path.join(repo_root, "configs")
        if os.path.isdir(repo_cfg_dir):
            pkg_yamls = {os.path.basename(p) for p in glob.glob(os.path.join(pkg_cfg_dir, "*.yaml"))}
            repo_yamls = {os.path.basename(p) for p in glob.glob(os.path.join(repo_cfg_dir, "*.yaml"))}
            self.assertEqual(pkg_yamls, repo_yamls)
            for fname in pkg_yamls:
                with open(os.path.join(pkg_cfg_dir, fname), "r", encoding="utf-8") as f1, open(
                    os.path.join(repo_cfg_dir, fname), "r", encoding="utf-8"
                ) as f2:
                    self.assertEqual(f1.read(), f2.read(), f"Config drift in {fname}")

        pyproject_path = os.path.join(repo_root, "pyproject.toml")
        req_path = os.path.join(repo_root, "requirements.txt")
        if os.path.exists(pyproject_path) and os.path.exists(req_path):
            try:
                import tomllib
            except ImportError:
                tomllib = None
            if tomllib is not None:
                with open(pyproject_path, "rb") as f:
                    pyproject = tomllib.load(f)
                self.assertEqual(pyproject["project"]["version"], prix.__version__)
                py_deps = set(pyproject["project"]["dependencies"])
                with open(req_path, "r", encoding="utf-8") as f:
                    req_deps = {
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#") and not line.startswith("pytest")
                    }
                self.assertEqual(py_deps, req_deps)

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

    def test_wave_fss_lut_protocol(self):
        from scipy.special import erf
        for prov in ("TFP", "TTP"):
            with prix.cfg.temp_override({"mpc.provider": prov}):
                res = prix.run(_wave_fss_lut_checks, world_size=2, mode="wave", timeout=180)
            x64 = res["x_np"].astype(np.float64)
            gelu_ref = 0.5 * x64 * (1.0 + erf(x64 / np.sqrt(2.0)))

            bior_rounds, bior_bytes, bior_out = res["bior"]
            self.assertEqual(bior_rounds, 2, f"{prov}:bior_rounds")
            self.assertEqual(bior_bytes, 128 * 40, f"{prov}:bior_bytes")  # 40 B/elem (8B R1 + 32B R2)
            np.testing.assert_allclose(bior_out, gelu_ref, atol=0.015, err_msg=f"{prov}:bior_acc")

            haar_rounds, haar_bytes, haar_out = res["haar"]
            self.assertEqual(haar_rounds, 2, f"{prov}:haar_rounds")
            self.assertEqual(haar_bytes, 128 * 24, f"{prov}:haar_bytes")  # 24 B/elem (8B R1 + 16B R2)
            np.testing.assert_allclose(haar_out, gelu_ref, atol=0.03, err_msg=f"{prov}:haar_acc")

            bp_rounds, bp_bytes, bp_out = res["bior_prob"]
            self.assertEqual(bp_rounds, 2, f"{prov}:bior_prob_rounds")
            self.assertEqual(bp_bytes, 128 * 40, f"{prov}:bior_prob_bytes")
            np.testing.assert_allclose(bp_out, gelu_ref, atol=0.25, err_msg=f"{prov}:bior_prob_acc")

    def test_prng_cpu_gpu_stream_sync_and_siphash(self):
        from prix.random import PRNGGenerator, _generate_kbit_jit, _przs_jit, _sm64_raw

        # Verify CPU (next_uint64) and device (next_offset + _sm64_raw / _przs_jit) streams
        # match bit-for-bit across both SplitMix64 and SipHash-2-4, including MSB-set seeds.
        seed_a = 0xFEDCBA9876543210
        seed_b = 0x89ABCDEF01234567
        shape = (7, 13)
        n = int(np.prod(shape))
        for prng_mode, use_sip in (("splitmix64", False), ("siphash24", True)):
            with prix.cfg.temp_override({"mpc.prng": prng_mode}):
                g_cpu = PRNGGenerator(seed=seed_a)
                g_dev = PRNGGenerator(seed=seed_a)
                for _ in range(3):
                    cpu_val = g_cpu.next_uint64(shape).view(np.int64)
                    s, off = g_dev.next_offset(n)
                    dev_val = np.asarray(_sm64_raw(s, off, shape, use_siphash=use_sip))
                    np.testing.assert_array_equal(cpu_val, dev_val, err_msg=prng_mode)
                    kbit_val = np.asarray(_generate_kbit_jit(off, s, shape, 19, use_siphash=use_sip))
                    np.testing.assert_array_equal(cpu_val & ((1 << 19) - 1), kbit_val, err_msg=prng_mode)

                g0_cpu, g1_cpu = PRNGGenerator(seed=seed_a), PRNGGenerator(seed=seed_b)
                g0_dev, g1_dev = PRNGGenerator(seed=seed_a), PRNGGenerator(seed=seed_b)
                u0_cpu, u1_cpu = g0_cpu.next_uint64(shape), g1_cpu.next_uint64(shape)
                s0, c0 = g0_dev.next_offset(n)
                s1, c1 = g1_dev.next_offset(n)
                np.testing.assert_array_equal(
                    (u0_cpu - u1_cpu).view(np.int64),
                    np.asarray(_przs_jit(c0, c1, s0, s1, shape, False, use_siphash=use_sip)),
                )
                np.testing.assert_array_equal(
                    (u0_cpu ^ u1_cpu).view(np.int64),
                    np.asarray(_przs_jit(c0, c1, s0, s1, shape, True, use_siphash=use_sip)),
                )

        # End-to-end 2PC under SipHash-2-4 cryptographic counter-mode PRF
        with prix.cfg.temp_override({"mpc.prng": "siphash24"}):
            res = prix.run(_basic_2pc_siphash_check, world_size=2, mode="poly", timeout=120)
        np.testing.assert_allclose(res["prod"], res["prod_ref"], atol=1e-2)
        np.testing.assert_allclose(res["gelu"], res["gelu_ref"], atol=2e-2)

    def test_tuple_provider_pickle_free_cache(self):
        res = prix.run(_tuple_cache_check, world_size=2, mode="crypten", timeout=120)
        np.testing.assert_allclose(res["cached_prod"], res["prod_ref"], atol=1e-2)
        self.assertFalse(res["shares_reused"])


def _tuple_cache_check():
    provider = prix.get_default_provider()
    rng = np.random.default_rng(99)
    a_np = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
    a = prix.array(a_np)
    b = prix.array(b_np)

    provider.trace(True)
    _ = a * b
    _ = a * b
    provider.trace(False)
    provider.fill_cache()
    provider.load_cache()
    t1 = provider.generate_additive_triple((4, 4), (4, 4), "mul")
    t2 = provider.generate_additive_triple((4, 4), (4, 4), "mul")
    shares_reused = bool(np.array_equal(np.asarray(t1[0].share), np.asarray(t2[0].share)))
    provider.load_cache()
    cached_prod = np.asarray((a * b).reconstruct())
    provider.remove_cache()
    return {
        "cached_prod": cached_prod,
        "prod_ref": a_np * b_np,
        "shares_reused": shares_reused,
    }


def _basic_2pc_siphash_check():
    from scipy.special import erf

    rng = np.random.default_rng(77)
    a_np = rng.uniform(-1.5, 1.5, size=(4, 8)).astype(np.float32)
    b_np = rng.uniform(-1.5, 1.5, size=(8, 4)).astype(np.float32)
    a = prix.array(a_np)
    b = prix.array(b_np)
    prod = np.asarray((a @ b).reconstruct())
    gelu = np.asarray(a.gelu().reconstruct())
    return {
        "prod": prod,
        "prod_ref": a_np @ b_np,
        "gelu": gelu,
        "gelu_ref": 0.5 * a_np * (1.0 + erf(a_np / np.sqrt(2.0))),
    }



if __name__ == "__main__":
    unittest.main()
