#!/usr/bin/env python3

import math
import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import prix
import prix.circuit as circuit
from prix.config import cfg
from prix.encoder import DATATYPE, FixedPointEncoder
from prix.launcher import MultiProcessLauncher
from prix.random import (
    PRNGGenerator,
    chebyshev_series,
    count_wraps,
    generate_kbit_random_tensor,
    generate_permutation,
    generate_random_ring_element,
)


def _run_mpc(fn, world_size=2, evaluator_size=0, cfg_file=None):
    launcher = MultiProcessLauncher(
        world_size, evaluator_size, fn, None, cfg_file=cfg_file
    )
    launcher.start()
    launcher.join()
    launcher.terminate()


def _worker_encoder_and_random():
    enc = FixedPointEncoder(precision_bits=16)
    vals = jnp.array([-3.5, 0.0, 1.25, 42.125], dtype=jnp.float32)
    encoded = enc.encode(vals)
    assert encoded.dtype == DATATYPE
    decoded = enc.decode(encoded)
    np.testing.assert_allclose(np.asarray(decoded), np.asarray(vals), atol=1e-4)

    g1 = PRNGGenerator(seed=12345)
    g2 = PRNGGenerator(seed=12345)
    r1 = generate_random_ring_element((4, 4), generator=g1)
    r2 = generate_random_ring_element((4, 4), generator=g2)
    assert jnp.array_equal(r1, r2)

    kbit = generate_kbit_random_tensor((100,), bitlength=8, generator=g1)
    assert jnp.all((kbit >= 0) & (kbit < 256))

    perm, inv_perm = generate_permutation(16, generator=g1)
    arr = jnp.arange(16)
    assert jnp.array_equal(arr[perm][inv_perm], arr)

    coeffs = chebyshev_series(jnp.tanh, 4.0, 10)
    assert coeffs.shape == (10,)


def _worker_arithmetic():
    x_val = jnp.array([[1.5, -2.0], [0.5, 3.25]], dtype=jnp.float32)
    y_val = jnp.array([[2.0, 4.0], [-1.0, 0.5]], dtype=jnp.float32)

    x = prix.cryptensor(x_val, src=0)
    y = prix.cryptensor(y_val, src=1)

    np.testing.assert_allclose(x.get_plain_text(), x_val, atol=1e-3)
    np.testing.assert_allclose((x + y).get_plain_text(), x_val + y_val, atol=1e-3)
    np.testing.assert_allclose((x - y).get_plain_text(), x_val - y_val, atol=1e-3)
    np.testing.assert_allclose((-x).get_plain_text(), -x_val, atol=1e-3)
    np.testing.assert_allclose((x * y).get_plain_text(), x_val * y_val, atol=1e-2)
    np.testing.assert_allclose((x @ y).get_plain_text(), x_val @ y_val, atol=1e-2)
    np.testing.assert_allclose(x.square().get_plain_text(), x_val**2, atol=1e-2)
    np.testing.assert_allclose((x / 2.0).get_plain_text(), x_val / 2.0, atol=1e-3)

    # Reductions & shape manipulations
    np.testing.assert_allclose(x.sum().get_plain_text(), jnp.sum(x_val), atol=1e-3)
    np.testing.assert_allclose(x.mean(dim=1).get_plain_text(), jnp.mean(x_val, axis=1), atol=1e-3)
    np.testing.assert_allclose(
        x.var(dim=1, unbiased=False).get_plain_text(), jnp.var(x_val, axis=1), atol=2e-2
    )
    np.testing.assert_allclose(x.cumsum(dim=0).get_plain_text(), jnp.cumsum(x_val, axis=0), atol=1e-3)
    np.testing.assert_allclose(x.trace().get_plain_text(), jnp.trace(x_val), atol=1e-3)
    np.testing.assert_allclose(
        x.roll(1, dims=0).get_plain_text(), jnp.roll(x_val, 1, axis=0), atol=1e-3
    )
    np.testing.assert_allclose(x.flip(0).get_plain_text(), jnp.flip(x_val, axis=0), atol=1e-3)

    cat_xy = prix.cat([x, y], dim=0)
    np.testing.assert_allclose(
        cat_xy.get_plain_text(), jnp.concatenate([x_val, y_val], axis=0), atol=1e-3
    )
    stack_xy = prix.stack([x, y], dim=0)
    np.testing.assert_allclose(
        stack_xy.get_plain_text(), jnp.stack([x_val, y_val], axis=0), atol=1e-3
    )
    s0, s1 = cat_xy.split(2, dim=0)
    np.testing.assert_allclose(s0.get_plain_text(), x_val, atol=1e-3)
    np.testing.assert_allclose(s1.get_plain_text(), y_val, atol=1e-3)


def _worker_binary_and_comparisons():
    a_val = jnp.array([-2.5, 0.0, 1.5, 3.0], dtype=jnp.float32)
    b_val = jnp.array([-1.0, 0.0, 2.0, 1.0], dtype=jnp.float32)

    a = prix.cryptensor(a_val)
    b = prix.cryptensor(b_val)

    # Binary conversions
    a_bin = a.to(prix.ptype.binary)
    a_back = a_bin.to(prix.ptype.arithmetic, precision=a.encoder.precision_bits)
    np.testing.assert_allclose(a_back.get_plain_text(), a_val, atol=1e-3)

    # Comparisons
    np.testing.assert_allclose((a < b).get_plain_text(), (a_val < b_val).astype(jnp.float32), atol=1e-3)
    np.testing.assert_allclose((a <= b).get_plain_text(), (a_val <= b_val).astype(jnp.float32), atol=1e-3)
    np.testing.assert_allclose((a > b).get_plain_text(), (a_val > b_val).astype(jnp.float32), atol=1e-3)
    np.testing.assert_allclose((a >= b).get_plain_text(), (a_val >= b_val).astype(jnp.float32), atol=1e-3)
    np.testing.assert_allclose((a == b).get_plain_text(), (a_val == b_val).astype(jnp.float32), atol=1e-3)
    np.testing.assert_allclose((a != b).get_plain_text(), (a_val != b_val).astype(jnp.float32), atol=1e-3)

    # Sign, abs, relu, hardtanh, where
    nz_val = jnp.array([-2.5, -0.5, 1.5, 3.0], dtype=jnp.float32)
    nz = prix.cryptensor(nz_val)
    np.testing.assert_allclose(nz.sign().get_plain_text(), jnp.sign(nz_val), atol=1e-3)
    np.testing.assert_allclose(nz.abs().get_plain_text(), jnp.abs(nz_val), atol=1e-2)
    np.testing.assert_allclose(nz.relu().get_plain_text(), jax.nn.relu(nz_val), atol=1e-2)
    np.testing.assert_allclose(
        nz.hardtanh(-1.0, 2.0).get_plain_text(), jnp.clip(nz_val, -1.0, 2.0), atol=2e-2
    )

    cond = a < b
    w = prix.where(cond, a, b)
    np.testing.assert_allclose(w.get_plain_text(), jnp.where(a_val < b_val, a_val, b_val), atol=2e-2)

    # Max, min, argmax, argmin
    mat_val = jnp.array([[1.0, 4.0, 2.0], [5.0, -1.0, 3.0]], dtype=jnp.float32)
    mat = prix.cryptensor(mat_val)
    np.testing.assert_allclose(mat.max().get_plain_text(), jnp.max(mat_val), atol=2e-2)
    np.testing.assert_allclose(mat.min().get_plain_text(), jnp.min(mat_val), atol=2e-2)
    max_row, _ = mat.max(dim=1)
    np.testing.assert_allclose(max_row.get_plain_text(), jnp.max(mat_val, axis=1), atol=2e-2)
    argmax_idx = mat.argmax(dim=1, one_hot=False)
    np.testing.assert_allclose(argmax_idx.get_plain_text(), jnp.argmax(mat_val, axis=1), atol=1e-3)


def _worker_approximations():
    pos_val = jnp.array([1.0, 1.5, 2.0, 3.0], dtype=jnp.float32)
    pos = prix.cryptensor(pos_val)

    inv_sqrt_val = jnp.array([0.25, 0.5, 1.0, 1.5], dtype=jnp.float32)
    inv_sqrt_in = prix.cryptensor(inv_sqrt_val)

    neg_val = jnp.array([-2.0, -1.5, -0.5, -0.1], dtype=jnp.float32)
    neg = prix.cryptensor(neg_val)

    x_val = jnp.array([-0.8, -0.4, 0.4, 0.8], dtype=jnp.float32)
    x = prix.cryptensor(x_val)

    with cfg.temp_override({"functions.exp.bior_size_bits": 8}):
        prix.LookupTables.LUTs.clear()
        np.testing.assert_allclose(neg.exp().get_plain_text(), jnp.exp(neg_val), rtol=0.1, atol=0.1)
        prix.LookupTables.LUTs.clear()
    with cfg.temp_override({"functions.exp.all_neg": False, "functions.exp.bior_size_bits": 8}):
        prix.LookupTables.LUTs.clear()
        np.testing.assert_allclose(x.exp().get_plain_text(), jnp.exp(x_val), rtol=0.15, atol=0.15)
        prix.LookupTables.LUTs.clear()
    np.testing.assert_allclose(pos.log().get_plain_text(), jnp.log(pos_val), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(pos.reciprocal().get_plain_text(), 1.0 / pos_val, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(pos.sqrt().get_plain_text(), jnp.sqrt(pos_val), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(
        inv_sqrt_in.inv_sqrt().get_plain_text(),
        1.0 / jnp.sqrt(inv_sqrt_val),
        rtol=0.1,
        atol=0.1,
    )
    np.testing.assert_allclose(x.sigmoid().get_plain_text(), jax.nn.sigmoid(x_val), rtol=0.1, atol=0.08)
    np.testing.assert_allclose(x.tanh().get_plain_text(), jnp.tanh(x_val), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(
        x.erf().get_plain_text(), jax.scipy.special.erf(x_val), rtol=0.1, atol=0.1
    )
    np.testing.assert_allclose(
        x.gelu().get_plain_text(), jax.nn.gelu(x_val, approximate=False), rtol=0.15, atol=0.1
    )
    with cfg.temp_override({"functions.silu.bior_size_bits": 7}):
        prix.LookupTables.LUTs.clear()
        np.testing.assert_allclose(x.silu().get_plain_text(), jax.nn.silu(x_val), rtol=0.1, atol=0.1)
        prix.LookupTables.LUTs.clear()
    np.testing.assert_allclose(x.sin().get_plain_text(), jnp.sin(x_val), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(x.cos().get_plain_text(), jnp.cos(x_val), rtol=0.1, atol=0.1)

    with cfg.temp_override({"functions.exp.bior_size_bits": 8}):
        prix.LookupTables.LUTs.clear()
        sm = x.softmax(dim=0).get_plain_text()
        prix.LookupTables.LUTs.clear()
    np.testing.assert_allclose(sm, jax.nn.softmax(x_val, axis=0), rtol=0.15, atol=0.08)

    ln_in_val = jnp.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], dtype=jnp.float32)
    ln_in = prix.cryptensor(ln_in_val)
    ln_out = ln_in.layernorm().get_plain_text()
    mean = jnp.mean(ln_in_val, axis=-1, keepdims=True)
    var = jnp.var(ln_in_val, axis=-1, keepdims=True)
    ln_ref = (ln_in_val - mean) / jnp.sqrt(var + 1e-5)
    np.testing.assert_allclose(ln_out, ln_ref, rtol=0.15, atol=0.15)


def _worker_fission():
    x_val = jnp.array([[-1.0, 0.5, 1.5, 2.0], [0.2, -0.5, 1.0, 0.8]], dtype=jnp.float32)
    x = prix.cryptensor(x_val)

    np.testing.assert_allclose(x.exp().get_plain_text(), jnp.exp(x_val), atol=2e-2)
    np.testing.assert_allclose(
        x.gelu().get_plain_text(), jax.nn.gelu(x_val, approximate=False), atol=2e-2
    )
    np.testing.assert_allclose(x.silu().get_plain_text(), jax.nn.silu(x_val), atol=2e-2)
    np.testing.assert_allclose(x.tanh().get_plain_text(), jnp.tanh(x_val), atol=2e-2)
    np.testing.assert_allclose(
        x.softmax(dim=-1).get_plain_text(), jax.nn.softmax(x_val, axis=-1), atol=2e-2
    )

    ln_out = x.layernorm().get_plain_text()
    mean = jnp.mean(x_val, axis=-1, keepdims=True)
    var = jnp.var(x_val, axis=-1, keepdims=True)
    ln_ref = (x_val - mean) / jnp.sqrt(var + 1e-5)
    np.testing.assert_allclose(ln_out, ln_ref, atol=5e-2)


def _worker_ttp():
    x_val = jnp.array([[1.5, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    y_val = jnp.array([[2.0, 1.0], [-1.0, 2.0]], dtype=jnp.float32)

    x = prix.cryptensor(x_val)
    y = prix.cryptensor(y_val)

    np.testing.assert_allclose((x * y).get_plain_text(), x_val * y_val, atol=1e-2)
    np.testing.assert_allclose((x @ y).get_plain_text(), x_val @ y_val, atol=1e-2)
    np.testing.assert_allclose(x.square().get_plain_text(), x_val**2, atol=1e-2)
    np.testing.assert_allclose(
        (x < y).get_plain_text(), (x_val < y_val).astype(jnp.float32), atol=1e-3
    )
    np.testing.assert_allclose(
        x.gelu().get_plain_text(), jax.nn.gelu(x_val, approximate=False), atol=0.1
    )
    with cfg.temp_override({"functions.gelu.method": "fourier"}):
        np.testing.assert_allclose(
            x.gelu().get_plain_text(), jax.nn.gelu(x_val, approximate=False), atol=5e-2
        )


def _worker_shaft_and_poly():
    x_val = jnp.array([-3.5, -1.5, -0.5, 0.0, 0.5, 1.5, 3.5], dtype=jnp.float32)
    x = prix.cryptensor(x_val)

    # SHAFT Fourier-series approximations
    with cfg.temp_override({"functions.gelu.method": "fourier"}):
        np.testing.assert_allclose(
            x.gelu().get_plain_text(), jax.nn.gelu(x_val, approximate=False), atol=3e-2
        )
    with cfg.temp_override({"functions.silu.method": "fourier"}):
        np.testing.assert_allclose(x.silu().get_plain_text(), jax.nn.silu(x_val), atol=4e-2)
    with cfg.temp_override({"functions.erf.method": "fourier"}):
        np.testing.assert_allclose(
            x.erf().get_plain_text(), jax.scipy.special.erf(x_val), atol=6e-2
        )
    with cfg.temp_override({"functions.sigmoid_tanh.method": "fourier"}):
        np.testing.assert_allclose(x.tanh().get_plain_text(), jnp.tanh(x_val), atol=6e-2)
        np.testing.assert_allclose(
            x.sigmoid().get_plain_text(), jax.nn.sigmoid(x_val), atol=6e-2
        )

    # Piecewise polynomial approximations
    with cfg.temp_override({"functions.gelu.method": "poly"}):
        np.testing.assert_allclose(
            x.gelu().get_plain_text(), jax.nn.gelu(x_val, approximate=False), atol=2e-2
        )
    with cfg.temp_override({"functions.silu.method": "poly"}):
        np.testing.assert_allclose(x.silu().get_plain_text(), jax.nn.silu(x_val), atol=3e-2)
    with cfg.temp_override({"functions.sigmoid_tanh.method": "poly"}):
        np.testing.assert_allclose(x.tanh().get_plain_text(), jnp.tanh(x_val), atol=2e-2)
        np.testing.assert_allclose(
            x.sigmoid().get_plain_text(), jax.nn.sigmoid(x_val), atol=2e-2
        )

    # SHAFT constant-round ODE softmax
    sm_val = jnp.array([[-1.0, 0.0, 1.0, 2.0], [0.5, 1.5, -0.5, 0.0]], dtype=jnp.float32)
    sm_in = prix.cryptensor(sm_val)
    with cfg.temp_override({"functions.softmax.method": "ode"}):
        sm_out = sm_in.softmax(dim=-1).get_plain_text()
    np.testing.assert_allclose(sm_out, jax.nn.softmax(sm_val, axis=-1), atol=5e-2)


def _worker_nn_and_llms():
    from examples.llms.models.bert import BertModel
    from examples.llms.models.gpt2 import GPT
    from examples.llms.models.llama import Llama
    from examples.llms.models.modern_bert import ModernBertModel

    # Basic nn layers
    lin = prix.nn.Linear(4, 3).encrypt()
    x = prix.cryptensor(jnp.ones((2, 4), dtype=jnp.float32) * 0.25)
    out = lin(x)
    assert out.shape == (2, 3)

    # Small BERT forward pass
    bert = BertModel(
        vocab_size=16, emb_size=8, seq_len=4, n_heads=2, n_layers=1, full=False
    ).encrypt()
    x_emb = prix.cryptensor(jnp.ones((1, 4, 8), dtype=jnp.float32) * 0.1)
    y_bert = bert(x_emb)
    assert y_bert.shape == (1, 4, 8)

    # Small GPT forward pass
    gpt = GPT(
        embed_dim=8, num_heads=2, num_blocks=1, vocab_size=16, seq_len=4, full=False
    ).encrypt()
    y_gpt = gpt(x_emb)
    assert y_gpt.shape == (1, 4, 8)

    # Small ModernBert forward pass
    mbert = ModernBertModel(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        seq_len=4,
        n_heads=2,
        n_layers=1,
        full=False,
    ).encrypt()
    y_mbert = mbert(x_emb)
    assert y_mbert.shape == (1, 4, 8)

    # Small Llama forward pass
    llama = Llama(
        embed_dim=8,
        head_dim=2,
        n_layers=1,
        n_kv_heads=2,
        vocab_size=16,
        multiple_of=4,
        ffn_dim_multiplier=1.0,
        seq_len=4,
        full=False,
        cache=True,
    ).encrypt()
    y_llama = llama(x_emb)
    assert y_llama.shape == (1, 4, 8)


def _worker_ergonomics(scale):
    z = prix.zeros(2, 3)
    o = prix.ones(2, 3)
    f = prix.full((2, 3), scale)
    a = prix.arange(3)
    e = prix.eye(3)
    np.testing.assert_allclose(z.get_plain_text(), np.zeros((2, 3)), atol=1e-3)
    np.testing.assert_allclose(o.get_plain_text(), np.ones((2, 3)), atol=1e-3)
    np.testing.assert_allclose(f.get_plain_text(), np.full((2, 3), scale), atol=1e-3)
    np.testing.assert_allclose(a.get_plain_text(), np.arange(3, dtype=np.float32), atol=1e-3)
    np.testing.assert_allclose(e.get_plain_text(), np.eye(3, dtype=np.float32), atol=1e-3)
    return (o + f).gelu().get_plain_text()


@prix.mpc.run_multiprocess(world_size=2, mode="shaft")
def _decorated_shaft_worker():
    x = prix.cryptensor([-1.0, 0.0, 1.0])
    return x.gelu().get_plain_text()


class TestPrix(unittest.TestCase):
    def test_01_encoder_and_random(self):
        _run_mpc(_worker_encoder_and_random)

    def test_02_arithmetic_ops(self):
        _run_mpc(_worker_arithmetic)

    def test_03_binary_and_comparisons(self):
        _run_mpc(_worker_binary_and_comparisons)

    def test_04_wavelet_luts(self):
        default_cfg = os.path.join(
            os.path.dirname(prix.__file__), "..", "configs", "default.yaml"
        )
        _run_mpc(_worker_approximations, cfg_file=default_cfg)

    def test_05_approximations(self):
        approx_cfg = os.path.join(
            os.path.dirname(prix.__file__), "..", "configs", "approximations.yaml"
        )
        _run_mpc(_worker_approximations, cfg_file=approx_cfg)

    def test_06_fission_evaluator(self):
        fission_cfg = os.path.join(
            os.path.dirname(prix.__file__), "..", "configs", "fission.yaml"
        )
        _run_mpc(_worker_fission, world_size=2, evaluator_size=1, cfg_file=fission_cfg)

    def test_07_ttp_provider(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            default_cfg = os.path.join(
                os.path.dirname(prix.__file__), "..", "configs", "default.yaml"
            )
            with open(default_cfg, "r") as src:
                content = src.read().replace("provider: \"TFP\"", "provider: \"TTP\"")
            f.write(content)
            ttp_cfg = f.name
        try:
            _run_mpc(_worker_ttp, world_size=2, evaluator_size=0, cfg_file=ttp_cfg)
        finally:
            if os.path.exists(ttp_cfg):
                os.remove(ttp_cfg)

    def test_08_nn_and_llms(self):
        fission_cfg = os.path.join(
            os.path.dirname(prix.__file__), "..", "configs", "fission.yaml"
        )
        _run_mpc(_worker_nn_and_llms, world_size=2, evaluator_size=1, cfg_file=fission_cfg)

    def test_09_shaft_and_poly(self):
        shaft_cfg = os.path.join(
            os.path.dirname(prix.__file__), "..", "configs", "shaft.yaml"
        )
        _run_mpc(_worker_shaft_and_poly, cfg_file=shaft_cfg)

    def test_10_researcher_ergonomics(self):
        for mode in ("crypten", "shaft", "lut", "fission"):
            prix.set_mode(mode)
        prix.set_mode("default")

        out_dec = _decorated_shaft_worker()
        # Verify prix.run with args and auto-evaluator in fission mode
        out_fission = prix.run(_worker_ergonomics, 2.5, world_size=2, mode="fission")
        prix.set_mode("default")

        # Compare against NumPy/erf reference without initializing JAX in the parent process
        def _np_gelu(x):
            x = np.asarray(x, dtype=np.float64)
            return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

        np.testing.assert_allclose(out_dec, _np_gelu([-1.0, 0.0, 1.0]), atol=5e-2)
        np.testing.assert_allclose(out_fission, _np_gelu(np.full((2, 3), 3.5)), atol=5e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
