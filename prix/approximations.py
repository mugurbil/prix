#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import functools
import logging
import math
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pywt
try:
    from scipy.special import erf as _scipy_erf
except ImportError:
    _scipy_erf = None

import prix
import prix.communicator as comm
from prix.config import cfg
from prix.encoder import DATATYPE, resolve_device, to_jax
from prix.evaluator import EvaluatorClient, check_supported as check_evaluator_supported
from prix.random import PRNGGenerator, _invert_permutation, chebyshev_series

__all__ = [
    "LookupTables",
    "permute_reveal_evaluate_share",
    "exp",
    "log",
    "reciprocal",
    "inv_sqrt",
    "sqrt",
    "_eix",
    "cossin",
    "cos",
    "sin",
    "sigmoid",
    "tanh",
    "erf",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "layernorm",
    "ge",
    "gt",
    "le",
    "lt",
    "eq",
    "ne",
    "sign",
    "abs",
    "relu",
    "hardtanh",
    "where",
    "argmax",
    "argmin",
    "max",
    "min",
    "pow",
    "pos_pow",
    "polynomial",
    "norm",
    "dropout",
    "weighted_index",
    "weighted_sample",
    "bernoulli",
    "randn",
    "max_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
]


class _LazyLUTDict(dict):
    def __getitem__(self, key):
        if LookupTables._initialized_precision != cfg.encoder.precision_bits:
            self.clear()
            LookupTables._initialized_precision = cfg.encoder.precision_bits
        if not super().__contains__(key):
            LookupTables._build_lut_by_name(key, device=LookupTables._default_device)
        return super().__getitem__(key)

    def __contains__(self, key):
        if LookupTables._initialized_precision != cfg.encoder.precision_bits:
            self.clear()
            LookupTables._initialized_precision = cfg.encoder.precision_bits
        return super().__contains__(key)


class LookupTables:
    """Singleton manager for wavelet-transformed Lookup Tables (Haar and Bior2.2)."""

    LUTs = _LazyLUTDict()
    _initialized_precision = None
    _default_device = None

    def __new__(cls, device=None, *args, **kwargs):
        it_id = "__it__"
        it = cls.__dict__.get(it_id, None)
        if it is not None:
            if cls._initialized_precision != cfg.encoder.precision_bits:
                cls.initialize_luts(device=device)
            return it
        it = object.__new__(cls)
        setattr(cls, it_id, it)
        it.init(*args, **kwargs)
        it.initialize_luts(device=device)
        return it

    def init(self, *args, **kwargs):
        pass

    @classmethod
    def _to_int64_np(cls, arr):
        arr_np = np.nan_to_num(
            np.asarray(arr, dtype=np.float64),
            nan=0.0,
            posinf=float(2**62 - 1),
            neginf=float(-(2**62)),
        )
        return np.clip(arr_np, float(-(2**62)), float(2**62 - 1)).astype(np.int64)

    @classmethod
    def _to_jax_lut(cls, arr, device=None):
        arr_i64 = jnp.asarray(cls._to_int64_np(arr), dtype=DATATYPE)
        if device is not None:
            arr_i64 = jax.device_put(arr_i64, resolve_device(device))
        return arr_i64

    @classmethod
    def _cache_path(cls, kind, name, max_bits, lut_bits, negative_values):
        cache_dir = os.path.join(tempfile.gettempdir(), "prix_lut_cache")
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"{kind}_{name}_p{cfg.encoder.precision_bits}_m{max_bits}_l{lut_bits}_n{int(negative_values)}.npy"
        return os.path.join(cache_dir, fname)

    @classmethod
    def generate_haar(cls, max_bits, lut_bits, function, name, negative_values=False, device=None):
        cpath = cls._cache_path("haar", name, max_bits, lut_bits, negative_values)
        if os.path.exists(cpath):
            try:
                arr_i64 = jnp.asarray(np.load(cpath), dtype=DATATYPE)
                if device is not None:
                    arr_i64 = jax.device_put(arr_i64, resolve_device(device))
                dict.__setitem__(cls.LUTs, name, arr_i64)
                return
            except Exception:
                pass
        scale = 2 ** cfg.encoder.precision_bits
        max_element = 2 ** max_bits
        depth = max_bits + cfg.encoder.precision_bits - lut_bits
        if negative_values:
            xs = np.linspace(-max_element + 1.0 / scale, max_element, int(2 * max_element * scale))
        else:
            xs = np.linspace(1.0 / scale, max_element, int(max_element * scale))
        full = function(xs)
        coeffs, *_ = pywt.wavedec(full, "haar", level=depth)
        arr_np = cls._to_int64_np(coeffs * (2 ** (-depth / 2.0)) * scale)
        try:
            tmp_p = f"{cpath}.{os.getpid()}.tmp.npy"
            np.save(tmp_p, arr_np)
            os.replace(tmp_p, cpath)
        except Exception:
            pass
        arr_i64 = jnp.asarray(arr_np, dtype=DATATYPE)
        if device is not None:
            arr_i64 = jax.device_put(arr_i64, resolve_device(device))
        dict.__setitem__(cls.LUTs, name, arr_i64)

    @classmethod
    def generate_bior(cls, max_bits, lut_bits, function, name, negative_values=False, device=None):
        cpath = cls._cache_path("bior", name, max_bits, lut_bits, negative_values)
        if os.path.exists(cpath):
            try:
                arr_i64 = jnp.asarray(np.load(cpath), dtype=DATATYPE)
                if device is not None:
                    arr_i64 = jax.device_put(arr_i64, resolve_device(device))
                dict.__setitem__(cls.LUTs, name, arr_i64)
                return
            except Exception:
                pass
        scale = 2 ** cfg.encoder.precision_bits
        max_element = 2 ** max_bits
        depth = max_bits + cfg.encoder.precision_bits - lut_bits
        if negative_values:
            xs = np.linspace(-max_element + 1.0 / scale, max_element, int(2 * max_element * scale))
            full = function(xs)
            coeffs, *_ = pywt.wavedec(full, "bior2.2", level=depth)
            coeffs = np.stack(
                [
                    np.roll(coeffs, -2)[: 2 ** (lut_bits + 1)],
                    np.roll(coeffs, -3)[: 2 ** (lut_bits + 1)],
                ]
            )
        else:
            xs = np.linspace(1.0 / scale, max_element, int(max_element * scale))
            full = function(xs)
            coeffs, *_ = pywt.wavedec(full, "bior2.2", level=depth)
            coeffs = np.stack(
                [
                    np.roll(coeffs, -2)[: 2 ** lut_bits],
                    np.roll(coeffs, -3)[: 2 ** lut_bits],
                ]
            )
        arr_np = cls._to_int64_np((coeffs * scale) * (2 ** (depth * 0.5)))
        try:
            tmp_p = f"{cpath}.{os.getpid()}.tmp.npy"
            np.save(tmp_p, arr_np)
            os.replace(tmp_p, cpath)
        except Exception:
            pass
        arr_i64 = jnp.asarray(arr_np, dtype=DATATYPE)
        if device is not None:
            arr_i64 = jax.device_put(arr_i64, resolve_device(device))
        dict.__setitem__(cls.LUTs, name, arr_i64)

    @classmethod
    def get_lut(cls, name, device=None):
        if cls._initialized_precision != cfg.encoder.precision_bits:
            cls.LUTs.clear()
            cls._initialized_precision = cfg.encoder.precision_bits
        if not dict.__contains__(cls.LUTs, name):
            cls._build_lut_by_name(name, device=device or cls._default_device)
        return dict.__getitem__(cls.LUTs, name)

    @classmethod
    def _build_lut_by_name(cls, name, device=None):
        scale = 2 ** cfg.encoder.precision_bits
        sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-x))
        relu_fn = lambda x: x * (x > 0)
        erf_fn = (
            _scipy_erf
            if _scipy_erf is not None
            else (lambda x: jax.scipy.special.erf(jnp.asarray(x, dtype=jnp.float64)).__array__())
        )
        gelu_fn = lambda x: x * (1.0 + erf_fn(x / math.sqrt(2.0))) / 2.0
        silu_fn = lambda x: x * sigmoid_fn(x)

        if name == "exp_haar":
            cls.generate_haar(
                cfg.functions.exp.lut_max_bits,
                cfg.functions.exp.haar_size_bits,
                np.exp,
                "exp_haar",
                negative_values=True,
                device=device,
            )
        elif name == "exp_bior":
            cls.generate_bior(
                cfg.functions.exp.lut_max_bits,
                cfg.functions.exp.bior_size_bits,
                np.exp,
                "exp_bior",
                negative_values=True,
                device=device,
            )
        elif name in ("nexp_low", "nexp_high"):
            size = cfg.functions.exp.neg_lut_size
            full_low = np.exp(-np.linspace(1.0 / size, 1.0 / (2**4), size))
            dict.__setitem__(cls.LUTs, "nexp_low", cls._to_jax_lut(full_low * scale, device=device))
            full_high = np.exp(-np.linspace(1.0 * (2**4) / size, 2**4, size))
            dict.__setitem__(cls.LUTs, "nexp_high", cls._to_jax_lut(full_high * scale, device=device))
        elif name == "nexp_haar":
            cls.generate_haar(
                cfg.functions.exp.lut_max_bits,
                cfg.functions.exp.haar_size_bits,
                lambda x: np.exp(-x),
                "nexp_haar",
                device=device,
            )
        elif name == "nexp_bior":
            cls.generate_bior(
                cfg.functions.exp.lut_max_bits,
                cfg.functions.exp.bior_size_bits,
                lambda x: np.exp(-x),
                "nexp_bior",
                device=device,
            )
        elif name == "log_haar":
            cls.generate_haar(
                cfg.functions.log.lut_max_bits,
                cfg.functions.log.haar_size_bits,
                np.log,
                "log_haar",
                device=device,
            )
        elif name == "log_bior":
            cls.generate_bior(
                cfg.functions.log.lut_max_bits,
                cfg.functions.log.bior_size_bits,
                np.log,
                "log_bior",
                device=device,
            )
        elif name == "reciprocal_haar":
            cls.generate_haar(
                cfg.functions.reciprocal.lut_max_bits,
                cfg.functions.reciprocal.haar_size_bits,
                np.reciprocal,
                "reciprocal_haar",
                device=device,
            )
        elif name == "reciprocal_bior":
            cls.generate_bior(
                cfg.functions.reciprocal.lut_max_bits,
                cfg.functions.reciprocal.bior_size_bits,
                np.reciprocal,
                "reciprocal_bior",
                device=device,
            )
        elif name == "sqrt_haar":
            cls.generate_haar(
                cfg.functions.sqrt.lut_max_bits,
                cfg.functions.sqrt.haar_size_bits,
                np.sqrt,
                "sqrt_haar",
                device=device,
            )
        elif name == "sqrt_bior":
            cls.generate_bior(
                cfg.functions.sqrt.lut_max_bits,
                cfg.functions.sqrt.bior_size_bits,
                np.sqrt,
                "sqrt_bior",
                device=device,
            )
        elif name == "inv_sqrt_haar":
            cls.generate_haar(
                cfg.functions.inv_sqrt.lut_max_bits,
                cfg.functions.inv_sqrt.haar_size_bits,
                lambda x: np.reciprocal(np.sqrt(x)),
                "inv_sqrt_haar",
                device=device,
            )
        elif name == "inv_sqrt_tailored_haar_0":
            cls.generate_haar(
                cfg.functions.inv_sqrt.tailored_0_lut_max_bits,
                cfg.functions.inv_sqrt.tailored_0_haar_size_bits,
                lambda x: np.reciprocal(np.sqrt(x)),
                "inv_sqrt_tailored_haar_0",
                device=device,
            )
        elif name == "inv_sqrt_tailored_haar_1":
            cls.generate_haar(
                cfg.functions.inv_sqrt.tailored_1_lut_max_bits,
                cfg.functions.inv_sqrt.tailored_1_haar_size_bits,
                lambda x: np.reciprocal(np.sqrt(x)),
                "inv_sqrt_tailored_haar_1",
                device=device,
            )
        elif name == "inv_sqrt_bior":
            cls.generate_bior(
                cfg.functions.inv_sqrt.lut_max_bits,
                cfg.functions.inv_sqrt.bior_size_bits,
                lambda x: np.reciprocal(np.sqrt(x)),
                "inv_sqrt_bior",
                device=device,
            )
        elif name == "sin_haar":
            cls.generate_haar(
                0,
                cfg.functions.trigonometry.haar_size_bits,
                lambda x: np.sin(x * np.pi * 2),
                "sin_haar",
                device=device,
            )
        elif name == "sin_bior":
            cls.generate_bior(
                0,
                cfg.functions.trigonometry.bior_size_bits,
                lambda x: np.sin(x * np.pi * 2),
                "sin_bior",
                device=device,
            )
        elif name == "sin_haar_lut_only":
            cls.generate_haar(
                cfg.functions.trigonometry.lut_max_bits,
                cfg.functions.trigonometry.haar_size_bits,
                lambda x: np.sin(x * np.pi * 2),
                "sin_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "sin_bior_lut_only":
            cls.generate_bior(
                cfg.functions.trigonometry.lut_max_bits,
                cfg.functions.trigonometry.bior_size_bits,
                lambda x: np.sin(x * np.pi * 2),
                "sin_bior_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "cos_haar":
            cls.generate_haar(
                0,
                cfg.functions.trigonometry.haar_size_bits,
                lambda x: np.cos(x * np.pi * 2),
                "cos_haar",
                device=device,
            )
        elif name == "cos_bior":
            cls.generate_bior(
                0,
                cfg.functions.trigonometry.bior_size_bits,
                lambda x: np.cos(x * np.pi * 2),
                "cos_bior",
                device=device,
            )
        elif name == "cos_haar_lut_only":
            cls.generate_haar(
                cfg.functions.trigonometry.lut_max_bits,
                cfg.functions.trigonometry.haar_size_bits,
                lambda x: np.cos(x * np.pi * 2),
                "cos_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "cos_bior_lut_only":
            cls.generate_bior(
                cfg.functions.trigonometry.lut_max_bits,
                cfg.functions.trigonometry.bior_size_bits,
                lambda x: np.cos(x * np.pi * 2),
                "cos_bior_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "sigmoid_haar":
            cls.generate_haar(
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits,
                cfg.functions.sigmoid_tanh.haar_size_bits,
                sigmoid_fn,
                "sigmoid_haar",
                device=device,
            )
        elif name == "sigmoid_bior":
            cls.generate_bior(
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits,
                cfg.functions.sigmoid_tanh.bior_size_bits,
                sigmoid_fn,
                "sigmoid_bior",
                device=device,
            )
        elif name == "sigmoid_haar_lut_only":
            cls.generate_haar(
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits,
                cfg.functions.sigmoid_tanh.haar_size_bits,
                sigmoid_fn,
                "sigmoid_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "sigmoid_bior_lut_only":
            cls.generate_bior(
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits,
                cfg.functions.sigmoid_tanh.bior_size_bits,
                sigmoid_fn,
                "sigmoid_bior_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "tanh_haar":
            cls.generate_haar(
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits,
                cfg.functions.sigmoid_tanh.haar_size_bits,
                np.tanh,
                "tanh_haar",
                device=device,
            )
        elif name == "tanh_bior":
            cls.generate_bior(
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits,
                cfg.functions.sigmoid_tanh.bior_size_bits,
                np.tanh,
                "tanh_bior",
                device=device,
            )
        elif name == "tanh_haar_lut_only":
            cls.generate_haar(
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits,
                cfg.functions.sigmoid_tanh.haar_size_bits,
                np.tanh,
                "tanh_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "tanh_bior_lut_only":
            cls.generate_bior(
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits,
                cfg.functions.sigmoid_tanh.bior_size_bits,
                np.tanh,
                "tanh_bior_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "erf_haar":
            cls.generate_haar(
                cfg.functions.erf.lut_max_bits,
                cfg.functions.erf.haar_size_bits,
                erf_fn,
                "erf_haar",
                device=device,
            )
        elif name == "erf_bior":
            cls.generate_bior(
                cfg.functions.erf.lut_max_bits,
                cfg.functions.erf.bior_size_bits,
                erf_fn,
                "erf_bior",
                device=device,
            )
        elif name == "erf_haar_lut_only":
            cls.generate_haar(
                cfg.functions.erf.lut_max_bits,
                cfg.functions.erf.haar_size_bits,
                erf_fn,
                "erf_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "erf_bior_lut_only":
            cls.generate_bior(
                cfg.functions.erf.lut_max_bits,
                cfg.functions.erf.bior_size_bits,
                erf_fn,
                "erf_bior_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "gelu_haar":
            cls.generate_haar(
                cfg.functions.gelu.lut_max_bits,
                cfg.functions.gelu.haar_size_bits,
                lambda x: relu_fn(x) - gelu_fn(x),
                "gelu_haar",
                device=device,
            )
        elif name == "gelu_bior":
            cls.generate_bior(
                cfg.functions.gelu.lut_max_bits,
                cfg.functions.gelu.bior_size_bits,
                lambda x: relu_fn(x) - gelu_fn(x),
                "gelu_bior",
                device=device,
            )
        elif name == "gelu_haar_lut_only":
            cls.generate_haar(
                cfg.functions.gelu.lut_max_bits,
                cfg.functions.gelu.haar_size_bits,
                gelu_fn,
                "gelu_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "gelu_bior_lut_only":
            cls.generate_bior(
                cfg.functions.gelu.lut_max_bits,
                cfg.functions.gelu.bior_size_bits,
                gelu_fn,
                "gelu_bior_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "silu_haar":
            cls.generate_haar(
                cfg.functions.silu.lut_max_bits,
                cfg.functions.silu.haar_size_bits,
                lambda x: relu_fn(x) - silu_fn(x),
                "silu_haar",
                device=device,
            )
        elif name == "silu_bior":
            cls.generate_bior(
                cfg.functions.silu.lut_max_bits,
                cfg.functions.silu.bior_size_bits,
                lambda x: relu_fn(x) - silu_fn(x),
                "silu_bior",
                device=device,
            )
        elif name == "silu_haar_lut_only":
            cls.generate_haar(
                cfg.functions.silu.lut_max_bits,
                cfg.functions.silu.haar_size_bits,
                silu_fn,
                "silu_haar_lut_only",
                negative_values=True,
                device=device,
            )
        elif name == "silu_bior_lut_only":
            cls.generate_bior(
                cfg.functions.silu.lut_max_bits,
                cfg.functions.silu.bior_size_bits,
                silu_fn,
                "silu_bior_lut_only",
                negative_values=True,
                device=device,
            )
        else:
            raise KeyError(f"Unknown LUT name: {name}")

    @classmethod
    def initialize_luts(cls, device=None, eager=False):
        """Initialize LUTs for configured approximation functions (lazy by default)."""
        cls._default_device = device
        if cls._initialized_precision != cfg.encoder.precision_bits:
            cls.LUTs.clear()
            cls._initialized_precision = cfg.encoder.precision_bits
        if not eager:
            return
        lut_methods = ("haar", "bior", "haar-lut-only", "bior-lut-only", "tailored_haar", "split")
        if cfg.functions.exp.method in lut_methods:
            cls._build_lut_by_name("exp_haar", device=device)
        if cfg.functions.log.method in lut_methods:
            cls._build_lut_by_name("log_haar", device=device)
        if cfg.functions.reciprocal.method in lut_methods:
            cls._build_lut_by_name("reciprocal_haar", device=device)
        if cfg.functions.sqrt.method in lut_methods:
            cls._build_lut_by_name("sqrt_haar", device=device)
        if cfg.functions.inv_sqrt.method in lut_methods:
            cls._build_lut_by_name("inv_sqrt_haar", device=device)
        if cfg.functions.trigonometry.method in lut_methods:
            cls._build_lut_by_name("sin_haar", device=device)
        if cfg.functions.sigmoid_tanh.method in lut_methods:
            cls._build_lut_by_name("sigmoid_haar", device=device)
        if cfg.functions.erf.method in lut_methods:
            cls._build_lut_by_name("erf_haar", device=device)
        if cfg.functions.gelu.method in lut_methods:
            cls._build_lut_by_name("gelu_haar", device=device)
        if cfg.functions.silu.method in lut_methods:
            cls._build_lut_by_name("silu_haar", device=device)


def _fresh_perm_np(n):
    """Fresh permutation shared by all MPC parties (jointly-seeded "global"
    generator) but unknown to the Fission evaluator. Never cached/reused."""
    import prix

    gen = prix.generators.get("global")
    if gen is None:
        raise RuntimeError("prix.init() must be called before Fission evaluation")
    perm = gen.next_permutation(int(n))
    return perm, _invert_permutation(perm)


def permute_reveal_evaluate_share(self, func, **extra):
    """Applies the permute and split reveal technique via Fission EvaluatorClient.

    `extra` holds plaintext function parameters (e.g. layernorm `eps`) that are
    forwarded to the evaluator in the request message.
    """
    check_evaluator_supported(func)
    shape = self.size()
    communicator = comm.get()
    if (
        communicator.get_evaluators_size() == 1
        and hasattr(communicator, "_put_to_mailbox")
        and cfg.mpc.provider == "TFP"
    ):
        arr_np = np.asarray(self.share)
        if func in ("layernorm", "softmax", "log_softmax") and arr_np.ndim > 1:
            flat_np = arr_np.reshape(-1, arr_np.shape[-1])
        else:
            flat_np = arr_np.reshape(-1)
        perm_np, inv_perm_np = _fresh_perm_np(flat_np.shape[0])
        shuffled_np = flat_np[perm_np]

        eval_rank = communicator.get_world_size() + (1 if communicator.ttp_initialized else 0)
        if communicator.get_rank() == 0:
            communicator.send_obj(
                {"function": func, "scale": self.scale, "array_size": shuffled_np.shape, **extra},
                eval_rank,
                EvaluatorClient.get().eval_comm_group,
            )
        communicator._put_to_mailbox(eval_rank, shuffled_np)
        _, recv_np = communicator._get_from_mailbox(src=eval_rank)
        out_np = recv_np[inv_perm_np].reshape(shape)
        out = self.shallow_copy()
        out.share = to_jax(out_np, device=self.device, dtype=DATATYPE)
        out.precision = cfg.encoder.precision_bits
        return out

    if func in ("layernorm", "softmax", "log_softmax") and self.dim() > 1:
        result = self.flatten(0, -2)
    else:
        result = self.flatten()
    result, inv_perm = result.shuffle()
    result = EvaluatorClient.get().evaluator_request(func, result, **extra)
    result = result.unshuffle(inv_perm)
    return result.reshape(shape)


def _nexp_lut(self, method):
    """Approximates the negative exponential function exp(-x) for x >= 0 using LUTs."""
    luts = LookupTables(self.device)
    precision = 2 ** cfg.encoder.precision_bits
    size = cfg.functions.exp.neg_lut_size

    if method == "split":
        x = self.div(precision / (2**4) / size)
        d = x < 1
        bits = x.precision
        x.precision = 0
        c = d * x + (1 - d) * (precision - 1)
        x.precision = bits
        c0 = c
        c1 = c.div(size)
        t0 = c0.evaluate_lut(luts.get_lut("nexp_low", self.device))
        t1 = c1.evaluate_lut(luts.get_lut("nexp_high", self.device))
        return t0 * t1
    elif method == "haar":
        check = self < (2 ** cfg.functions.exp.lut_max_bits)
        truncation = (
            cfg.functions.exp.lut_max_bits
            + cfg.encoder.precision_bits
            - cfg.functions.exp.haar_size_bits
        )
        if cfg.encoder.trunc_method.lut == "crypten":
            msb = self.div(2 ** truncation)
        else:
            msb = self.egk_trunc_pr(62, truncation)
        lut = msb.evaluate_lut(luts.get_lut("nexp_haar", self.device))
        return check * lut
    elif method == "bior":
        check = self < (2 ** cfg.functions.exp.lut_max_bits)
        truncation = (
            cfg.functions.exp.lut_max_bits
            + cfg.encoder.precision_bits
            - cfg.functions.exp.bior_size_bits
        )
        if cfg.encoder.trunc_method.lut == "crypten":
            msb, lsb = self.divmod(2 ** truncation)
        else:
            msb, lsb = self.egk_truncmod_pr(62, truncation)
        lut = msb.evaluate_bior_lut(luts.get_lut("nexp_bior", self.device), lsb, truncation)
        return check * lut
    else:
        raise ValueError(f"Invalid method {method} given for nexp function")


def exp(self):
    """Approximates the exponential function."""
    method = cfg.functions.exp.method

    if method in ("split", "haar", "bior"):
        if cfg.functions.exp.all_neg:
            return _nexp_lut(-self, method)
        if method == "split":
            raise ValueError(
                "exp method 'split' only supports non-positive inputs; "
                "set cfg.functions.exp.all_neg=True or use 'haar'/'bior'"
            )
        luts = LookupTables(self.device)
        shifted = self + (2 ** cfg.functions.exp.lut_max_bits)
        if method == "haar":
            truncation = (
                cfg.functions.exp.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.exp.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = shifted.div(2 ** truncation)
            else:
                msb = shifted.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("exp_haar", self.device))
        elif method == "bior":
            truncation = (
                cfg.functions.exp.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.exp.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = shifted.divmod(2 ** truncation)
            else:
                msb, lsb = shifted.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(luts.get_lut("exp_bior", self.device), lsb, truncation)
    elif method == "limit":
        iters = cfg.functions.exp.iterations
        result = 1 + self.div(2 ** iters)
        for _ in range(iters):
            result = result.square()
        return result
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "exp")
    else:
        raise ValueError(f"Invalid method {method} given for exp function")


def log(self, input_in_01=False):
    """Approximates the natural logarithm."""
    if input_in_01:
        return log(self.mul(100)) - 4.605170

    iterations = cfg.functions.log.iterations
    exp_iterations = cfg.functions.log.exp_iterations
    order = cfg.functions.log.order
    method = cfg.functions.log.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        if method == "haar":
            log_truncation = (
                cfg.functions.log.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.log.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = self.div(2 ** log_truncation)
            else:
                msb = self.egk_trunc_pr(62, log_truncation)
            return msb.evaluate_lut(luts.get_lut("log_haar", self.device))
        elif method == "bior":
            log_truncation = (
                cfg.functions.log.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.log.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = self.divmod(2 ** log_truncation)
            else:
                msb, lsb = self.egk_truncmod_pr(62, log_truncation)
            return msb.evaluate_bior_lut(luts.get_lut("log_bior", self.device), lsb, log_truncation)
    elif method == "iter":
        term1 = self.div(120)
        term2 = exp(self.mul(2).add(1.0).neg()).mul(20)
        y = term1 - term2 + 3.0

        with cfg.temp_override({"functions.exp.iterations": exp_iterations, "functions.exp.method": "limit"}):
            for _ in range(iterations):
                h = 1 - self * exp(-y)
                y = y - h.polynomial([1 / (i + 1) for i in range(order)])
        return y
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "log")
    else:
        raise ValueError(f"Invalid method {method} given for log function")


def reciprocal(self, input_in_01=False):
    """Computes the reciprocal of the input tensor."""
    pos_override = {"functions.reciprocal.all_pos": True}
    if input_in_01:
        with cfg.temp_override(pos_override):
            rec = reciprocal(self.mul(64)).mul(64)
        return rec

    method = cfg.functions.reciprocal.method
    all_pos = cfg.functions.reciprocal.all_pos
    initial = cfg.functions.reciprocal.initial

    if not all_pos and method != "fission":
        sgn = self.sign()
        pos = sgn * self
        with cfg.temp_override(pos_override):
            return sgn * reciprocal(pos)

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        if method == "haar":
            reciprocal_truncation = (
                cfg.functions.reciprocal.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.reciprocal.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = self.div(2 ** reciprocal_truncation)
            else:
                msb = self.egk_trunc_pr(62, reciprocal_truncation)
            return msb.evaluate_lut(luts.get_lut("reciprocal_haar", self.device))
        elif method == "bior":
            reciprocal_truncation = (
                cfg.functions.reciprocal.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.reciprocal.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = self.divmod(2 ** reciprocal_truncation)
            else:
                msb, lsb = self.egk_truncmod_pr(62, reciprocal_truncation)
            return msb.evaluate_bior_lut(
                luts.get_lut("reciprocal_bior", self.device), lsb, reciprocal_truncation
            )
    elif method == "NR":
        nr_iters = cfg.functions.reciprocal.nr_iters
        if initial is None:
            with cfg.temp_override({"functions.exp.method": "limit"}):
                result = 3 * (1 - 2 * self).exp() + 0.003
        else:
            result = initial
        for _ in range(nr_iters):
            if hasattr(result, "square"):
                result = result + result - result.square().mul(self)
            else:
                result = 2 * result - result * result * self
        return result
    elif method == "log":
        log_iters = cfg.functions.reciprocal.log_iters
        with cfg.temp_override({"functions.log.iterations": log_iters}):
            return exp(-log(self))
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "reciprocal")
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(self):
    """Computes the inverse square root of the input."""
    method = cfg.functions.inv_sqrt.method

    if method in ("haar", "bior", "tailored_haar"):
        luts = LookupTables(self.device)
        if method == "haar":
            truncation = (
                cfg.functions.inv_sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.inv_sqrt.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = self.div(2 ** truncation)
            else:
                msb = self.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("inv_sqrt_haar", self.device))
        elif method == "bior":
            truncation = (
                cfg.functions.inv_sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.inv_sqrt.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = self.divmod(2 ** truncation)
            else:
                msb, lsb = self.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(luts.get_lut("inv_sqrt_bior", self.device), lsb, truncation)
        elif method == "tailored_haar":
            truncation_0 = (
                cfg.functions.inv_sqrt.tailored_0_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.inv_sqrt.tailored_0_haar_size_bits
            )
            truncation_1 = (
                cfg.functions.inv_sqrt.tailored_1_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.inv_sqrt.tailored_1_haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb_0 = self.div(2 ** truncation_0)
                msb_1 = self.div(2 ** truncation_1)
            else:
                msb_0 = self.egk_trunc_pr(62, truncation_0)
                msb_1 = self.egk_trunc_pr(62, truncation_1)
            y_0 = msb_0.evaluate_lut(luts.get_lut("inv_sqrt_tailored_haar_0", self.device))
            y_1 = msb_1.evaluate_lut(luts.get_lut("inv_sqrt_tailored_haar_1", self.device))
            b = self < 1
            return b * y_0 + (1 - b) * y_1
    elif method == "NR":
        # Prefer inv_sqrt-specific NR settings; fall back to the shared sqrt ones.
        initial = getattr(cfg.functions.inv_sqrt, "nr_initial", cfg.functions.sqrt.nr_initial)
        iters = getattr(cfg.functions.inv_sqrt, "nr_iters", cfg.functions.sqrt.nr_iters)
        if initial is None:
            with cfg.temp_override({"functions.exp.method": "limit"}):
                y = exp(self.div(2).add(0.2).neg()).mul(2.2).add(0.2)
            y = y - self.div(1024)
        else:
            y = initial

        for _ in range(iters):
            y = y.mul(3 - self * y.square()).div(2)
        return y
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "inv_sqrt")
    else:
        raise ValueError(f"Invalid method {method} given for inv_sqrt function")


def sqrt(self):
    """Computes the square root of the input."""
    method = cfg.functions.sqrt.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        if method == "haar":
            truncation = (
                cfg.functions.sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sqrt.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = self.div(2 ** truncation)
            else:
                msb = self.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("sqrt_haar", self.device))
        elif method == "bior":
            truncation = (
                cfg.functions.sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sqrt.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = self.divmod(2 ** truncation)
            else:
                msb, lsb = self.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(luts.get_lut("sqrt_bior", self.device), lsb, truncation)
    elif method == "NR":
        with cfg.temp_override({"functions.inv_sqrt.method": "NR"}):
            return inv_sqrt(self).mul(self)
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "sqrt")
    else:
        raise ValueError(f"Invalid method {method} given for sqrt function")


def _eix(self):
    """Computes e^(i * self) = (cos(self), sin(self))."""
    iterations = cfg.functions.trigonometry.iterations

    re = 1
    im = self.div(2 ** iterations)

    re = re - im.square()
    im = im * 2

    for _ in range(iterations - 1):
        a2 = re.square()
        b2 = im.square()
        im = im.mul(re) * 2
        re = a2 - b2

    return re, im


def cossin(self):
    """Computes cosine and sine of input."""
    method = cfg.functions.trigonometry.method
    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        sgn = self.sign()
        x = sgn * self
        x = x * (1.0 / (2 * np.pi))
        x = x.mod(2 ** cfg.encoder.precision_bits)
        if method == "haar":
            trig_truncation = cfg.encoder.precision_bits - cfg.functions.trigonometry.haar_size_bits
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** trig_truncation)
            else:
                msb = x.egk_trunc_pr(62, trig_truncation)
            cos_val = msb.evaluate_lut(luts.get_lut("cos_haar", self.device))
            sin_val = msb.evaluate_lut(luts.get_lut("sin_haar", self.device))
        elif method == "bior":
            trig_truncation = cfg.encoder.precision_bits - cfg.functions.trigonometry.bior_size_bits
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** trig_truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, trig_truncation)
            cos_val = msb.evaluate_bior_lut(
                luts.get_lut("cos_bior", self.device), lsb, trig_truncation
            )
            sin_val = msb.evaluate_bior_lut(
                luts.get_lut("sin_bior", self.device), lsb, trig_truncation
            )
        sin_val = sgn * sin_val
        return cos_val, sin_val
    elif method in ("haar-lut-only", "bior-lut-only"):
        luts = LookupTables(self.device)
        x = self + (2 ** cfg.functions.trigonometry.lut_max_bits)
        if method == "haar-lut-only":
            truncation = (
                cfg.functions.trigonometry.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.trigonometry.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** truncation)
            else:
                msb = x.egk_trunc_pr(62, truncation)
            cos_val = msb.evaluate_lut(luts.get_lut("cos_haar_lut_only", self.device))
            sin_val = msb.evaluate_lut(luts.get_lut("sin_haar_lut_only", self.device))
            return cos_val, sin_val
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.trigonometry.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.trigonometry.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, truncation)
            cos_val = msb.evaluate_bior_lut(
                luts.get_lut("cos_bior_lut_only", self.device), lsb, truncation
            )
            sin_val = msb.evaluate_bior_lut(
                luts.get_lut("sin_bior_lut_only", self.device), lsb, truncation
            )
            return cos_val, sin_val
    elif method == "NR":
        return self._eix()
    elif method == "fission":
        return (
            permute_reveal_evaluate_share(self, "cos"),
            permute_reveal_evaluate_share(self, "sin"),
        )
    else:
        raise ValueError(f"Invalid method {method} given for cossin function")


def cos(self):
    """Computes the cosine of the input."""
    return cossin(self)[0]


def sin(self):
    """Computes the sine of the input."""
    return cossin(self)[1]


def sigmoid(self):
    """Computes the sigmoid function."""
    method = cfg.functions.sigmoid_tanh.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        ltz = self._ltz()
        sgn = 1 - 2 * ltz
        abs_val = sgn * self
        if method == "haar":
            st_truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = abs_val.div(2 ** st_truncation)
            else:
                msb = abs_val.egk_trunc_pr(62, st_truncation)
            lut = msb.evaluate_lut(luts.get_lut("sigmoid_haar", self.device))
        elif method == "bior":
            st_truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = abs_val.divmod(2 ** st_truncation)
            else:
                msb, lsb = abs_val.egk_truncmod_pr(62, st_truncation)
            lut = msb.evaluate_bior_lut(
                luts.get_lut("sigmoid_bior", self.device), lsb, st_truncation
            )
        eval_val = ltz + sgn * lut
        limit = 1 - ltz
        check = abs_val < (2 ** cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits - 1)
        return limit + check * (eval_val - limit)
    elif method in ("haar-lut-only", "bior-lut-only"):
        luts = LookupTables(self.device)
        x = self + (2 ** cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits)
        if method == "haar-lut-only":
            truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** truncation)
            else:
                msb = x.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("sigmoid_haar_lut_only", self.device))
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(
                luts.get_lut("sigmoid_bior_lut_only", self.device), lsb, truncation
            )
    elif method == "chebyshev":
        tanh_approx = tanh(self.div(2))
        return tanh_approx.div(2) + 0.5
    elif method == "reciprocal":
        ltz = self._ltz()
        sgn = 1 - 2 * ltz

        pos_input = self.mul(sgn)
        with cfg.temp_override(
            {
                "functions.exp.method": "limit",
                "functions.exp.iterations": 9,
                "functions.reciprocal.method": "NR",
                "functions.reciprocal.nr_iters": 3,
                "functions.reciprocal.all_pos": True,
                "functions.reciprocal.initial": 0.75,
            }
        ):
            denominator = pos_input.neg().exp().add(1)
            pos_output = denominator.reciprocal()

        return pos_output.where(1 - ltz, 1 - pos_output)
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "sigmoid")
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")


def _chebyshev_polynomials(self, terms):
    """Evaluates odd degree Chebyshev polynomials at x."""
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [self.clone()]
    y = 4 * self.square() - 2
    z = y - 1
    polynomials.append(z.mul(self))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return prix.stack(polynomials)


def tanh(self):
    """Computes the hyperbolic tangent function."""
    method = cfg.functions.sigmoid_tanh.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        sgn = self.sign()
        abs_val = sgn * self
        if method == "haar":
            st_truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = abs_val.div(2 ** st_truncation)
            else:
                msb = abs_val.egk_trunc_pr(62, st_truncation)
            lut = msb.evaluate_lut(luts.get_lut("tanh_haar", self.device))
        elif method == "bior":
            st_truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = abs_val.divmod(2 ** st_truncation)
            else:
                msb, lsb = abs_val.egk_truncmod_pr(62, st_truncation)
            lut = msb.evaluate_bior_lut(luts.get_lut("tanh_bior", self.device), lsb, st_truncation)
        check = abs_val < (2 ** cfg.functions.sigmoid_tanh.tanh_lut_max_bits - 1)
        return sgn * (1 - check + lut * check)
    elif method in ("haar-lut-only", "bior-lut-only"):
        luts = LookupTables(self.device)
        x = self + (2 ** cfg.functions.sigmoid_tanh.tanh_lut_max_bits)
        if method == "haar-lut-only":
            truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** truncation)
            else:
                msb = x.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("tanh_haar_lut_only", self.device))
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(
                luts.get_lut("tanh_bior_lut_only", self.device), lsb, truncation
            )
    elif method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh.terms
        coeffs = chebyshev_series(np.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        out = tanh_polys_flipped.matmul(coeffs)
        return out.hardtanh()
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "tanh")
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")


def erf(self):
    """Approximates the error function of the input tensor."""
    method = cfg.functions.erf.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        sgn = self.sign()
        abs_val = sgn * self
        if method == "haar":
            erf_truncation = (
                cfg.functions.erf.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.erf.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = abs_val.div(2 ** erf_truncation)
            else:
                msb = abs_val.egk_trunc_pr(62, erf_truncation)
            lut = msb.evaluate_lut(luts.get_lut("erf_haar", self.device))
        elif method == "bior":
            erf_truncation = (
                cfg.functions.erf.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.erf.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = abs_val.divmod(2 ** erf_truncation)
            else:
                msb, lsb = abs_val.egk_truncmod_pr(62, erf_truncation)
            lut = msb.evaluate_bior_lut(luts.get_lut("erf_bior", self.device), lsb, erf_truncation)
        check = abs_val < (2 ** cfg.functions.erf.lut_max_bits - 1)
        return sgn * (1 - check + lut * check)
    elif method in ("haar-lut-only", "bior-lut-only"):
        luts = LookupTables(self.device)
        x = self + (2 ** cfg.functions.erf.lut_max_bits)
        if method == "haar-lut-only":
            truncation = (
                cfg.functions.erf.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.erf.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** truncation)
            else:
                msb = x.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("erf_haar_lut_only", self.device))
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.erf.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.erf.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(
                luts.get_lut("erf_bior_lut_only", self.device), lsb, truncation
            )
    elif method == "Taylor":
        iters = cfg.functions.erf.iterations
        output = self.clone()
        for n in range(1, iters + 1):
            multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
            output = output.add(self.pos_pow(2 * n + 1).mul(multiplier))
        return output.mul(2.0 / math.sqrt(math.pi))
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "erf")
    else:
        raise ValueError(f"Unrecognized method {method} for erf")


def gelu(self):
    """Approximates the GELU function of the input tensor."""
    method = cfg.functions.gelu.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        # One comparison + one multiplication for both |x| and relu(x):
        # with m = x * [x < 0]:  |x| = x - 2m,  relu(x) = x - m.
        neg_part = self * self._ltz()
        abs_val = self - 2 * neg_part
        relu_val = self - neg_part
        if method == "haar":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = abs_val.div(2 ** truncation)
            else:
                msb = abs_val.egk_trunc_pr(62, truncation)
            lut = msb.evaluate_lut(luts.get_lut("gelu_haar", self.device))
        elif method == "bior":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = abs_val.divmod(2 ** truncation)
            else:
                msb, lsb = abs_val.egk_truncmod_pr(62, truncation)
            lut = msb.evaluate_bior_lut(luts.get_lut("gelu_bior", self.device), lsb, truncation)
        check = abs_val < (2 ** cfg.functions.gelu.lut_max_bits)
        return relu_val - lut * check
    elif method in ("haar-lut-only", "bior-lut-only"):
        luts = LookupTables(self.device)
        x = self + (2 ** cfg.functions.gelu.lut_max_bits)
        if method == "haar-lut-only":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** truncation)
            else:
                msb = x.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("gelu_haar_lut_only", self.device))
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(
                luts.get_lut("gelu_bior_lut_only", self.device), lsb, truncation
            )
    elif method == "erf":
        return self * (1 + (self / math.sqrt(2)).erf()) / 2
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "gelu")
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")


def silu(self):
    """Approximates the SiLU (Swish) function of the input tensor."""
    method = cfg.functions.silu.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        # One comparison + one multiplication for both |x| and relu(x):
        # with m = x * [x < 0]:  |x| = x - 2m,  relu(x) = x - m.
        neg_part = self * self._ltz()
        abs_val = self - 2 * neg_part
        relu_val = self - neg_part
        if method == "haar":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = abs_val.div(2 ** truncation)
            else:
                msb = abs_val.egk_trunc_pr(62, truncation)
            lut = msb.evaluate_lut(luts.get_lut("silu_haar", self.device))
        elif method == "bior":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = abs_val.divmod(2 ** truncation)
            else:
                msb, lsb = abs_val.egk_truncmod_pr(62, truncation)
            lut = msb.evaluate_bior_lut(luts.get_lut("silu_bior", self.device), lsb, truncation)
        check = abs_val < (2 ** cfg.functions.silu.lut_max_bits - 1)
        return relu_val - lut * check
    elif method in ("haar-lut-only", "bior-lut-only"):
        luts = LookupTables(self.device)
        x = self + (2 ** cfg.functions.silu.lut_max_bits)
        if method == "haar-lut-only":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.haar_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb = x.div(2 ** truncation)
            else:
                msb = x.egk_trunc_pr(62, truncation)
            return msb.evaluate_lut(luts.get_lut("silu_haar_lut_only", self.device))
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.bior_size_bits
            )
            if cfg.encoder.trunc_method.lut == "crypten":
                msb, lsb = x.divmod(2 ** truncation)
            else:
                msb, lsb = x.egk_truncmod_pr(62, truncation)
            return msb.evaluate_bior_lut(
                luts.get_lut("silu_bior_lut_only", self.device), lsb, truncation
            )
    elif method == "sigmoid":
        return self * self.sigmoid()
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "silu")
    else:
        raise ValueError(f"Unrecognized method {method} for silu")


def softmax(self, dim=-1, **kwargs):
    """Computes the softmax of a tensor's elements along a given dimension."""
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(jnp.ones_like(self.share, dtype=jnp.float32))

    if self.size(dim) == 1:
        return self.new(jnp.ones_like(self.share, dtype=jnp.float32))

    method = cfg.functions.softmax.method
    if method == "approximation":
        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        with cfg.temp_override({"functions.exp.all_neg": True}):
            numerator = logits.exp()
        with cfg.temp_override({"functions.reciprocal.all_pos": True}):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
    elif method == "fission":
        if dim not in (-1, self.dim() - 1):
            raise ValueError(f"Dimension {dim} not supported for fission softmax")
        return permute_reveal_evaluate_share(self, "softmax")
    else:
        raise ValueError(f"Unrecognized method {method} for softmax")


def log_softmax(self, dim=-1, **kwargs):
    """Applies a softmax followed by a logarithm."""
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(jnp.zeros((), dtype=jnp.float32))

    if self.size(dim) == 1:
        return self.new(jnp.zeros_like(self.share, dtype=jnp.float32))

    method = cfg.functions.log_softmax.method
    if method == "approximation":
        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        with cfg.temp_override({"functions.exp.all_neg": True}):
            normalize_term = logits.exp().sum(dim, keepdim=True)
        return logits - normalize_term.log()
    elif method == "fission":
        if dim not in (-1, self.dim() - 1):
            raise ValueError(f"Dimension {dim} not supported for fission log_softmax")
        return permute_reveal_evaluate_share(self, "log_softmax")
    else:
        raise ValueError(f"Unrecognized method {method} for log_softmax")


def layernorm(self, weight=None, bias=None, eps=1e-5):
    """Computes layer normalization over the last dimension."""
    ln_cfg = getattr(cfg.functions, "layernorm", None)
    method = ln_cfg.method if ln_cfg is not None else "approximation"
    if method == "approximation":
        mean = self.mean(-1, keepdim=True)
        diff = self - mean
        variance = diff.square().mean(-1, keepdim=True)
        inv_std = (variance + eps).inv_sqrt()
        out = diff * inv_std
    elif method == "fission":
        out = permute_reveal_evaluate_share(self, "layernorm", eps=float(eps))
    else:
        raise ValueError(f"Unrecognized method {method} for layernorm")

    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


# Logic and comparison functions
def ge(self, y):
    """Returns self >= y."""
    return 1 - self.lt(y)


def gt(self, y):
    """Returns self > y."""
    return (-self + y)._ltz()


def le(self, y):
    """Returns self <= y."""
    return 1 - self.gt(y)


def lt(self, y):
    """Returns self < y."""
    return (self - y)._ltz()


def eq(self, y):
    """Returns self == y."""
    if comm.get().get_world_size() == 2:
        return (self - y)._eqz_2PC()
    return 1 - self.ne(y)


def ne(self, y):
    """Returns self != y."""
    if comm.get().get_world_size() == 2:
        return 1 - self.eq(y)
    difference = self - y
    difference = type(difference).stack([difference, -difference])
    return difference._ltz().sum(0)


def sign(self):
    """Computes the sign value of a tensor (0 is considered positive: +1)."""
    return 1 - 2 * self._ltz()


def abs(self):
    """Computes the absolute value of a tensor."""
    return self * self.sign()


def relu(self):
    """Computes a Rectified Linear function on the input tensor."""
    method = getattr(cfg.functions, "relu", None)
    if method is not None and getattr(method, "method", None) == "fission":
        return permute_reveal_evaluate_share(self, "relu")
    return self * self.ge(0)


def hardtanh(self, min_value=-1, max_value=1):
    """Applies the HardTanh function element-wise."""
    intermediate = prix.stack([self - min_value, self - max_value]).relu()
    intermediate = intermediate[0].sub(intermediate[1])
    return intermediate.add(min_value)


def where(self, condition, y):
    """Selects elements from self or y based on condition."""
    if isinstance(condition, prix.Array):
        y_masked = (1 - condition) * y
        return self * condition + y_masked
    else:
        cond_arr = jnp.asarray(condition, dtype=jnp.float32)
        y_masked = y * (1.0 - cond_arr)
        return self * cond_arr + y_masked


# Maximum / Argmax functions
def argmax(self, dim=None, keepdim=False, one_hot=True):
    """Returns the indices of the maximum value of all elements in the input tensor."""
    method = cfg.functions.max.method

    if self.dim() == 0:
        return (
            self.new(jnp.ones((), dtype=jnp.float32))
            if one_hot
            else self.new(jnp.zeros((), dtype=jnp.float32))
        )

    result = _argmax_helper(self, dim, one_hot, method, _return_max=False)
    if not one_hot:
        result = _one_hot_to_index(result, dim, keepdim, self.device)
    return result


def argmin(self, dim=None, keepdim=False, one_hot=True):
    """Returns the indices of the minimum value of all elements in the input tensor."""
    return (-self).argmax(dim=dim, keepdim=keepdim, one_hot=one_hot)


def max(self, dim=None, keepdim=False, one_hot=True):
    """Returns the maximum value of all elements in the input tensor."""
    method = cfg.functions.max.method
    if dim is None:
        if method in ["log_reduction", "double_log_reduction"]:
            max_result = _max_helper_all_tree_reductions(self, method=method)
        else:
            with cfg.temp_override({"functions.max.method": method}):
                argmax_result = self.argmax(one_hot=True)
            max_result = self.mul(argmax_result).sum()
        return max_result
    else:
        argmax_result, max_result = _argmax_helper(
            self, dim=dim, one_hot=True, method=method, _return_max=True
        )
        if max_result is None:
            max_result = (self * argmax_result).sum(dim=dim, keepdim=keepdim)
        if keepdim:
            max_result = (
                max_result.unsqueeze(dim) if max_result.dim() < self.dim() else max_result
            )
        if one_hot:
            return max_result, argmax_result
        else:
            return (
                max_result,
                _one_hot_to_index(argmax_result, dim, keepdim, self.device),
            )


def min(self, dim=None, keepdim=False, one_hot=True):
    """Returns the minimum value of all elements in the input tensor."""
    result = (-self).max(dim=dim, keepdim=keepdim, one_hot=one_hot)
    if dim is None:
        return -result
    else:
        return -result[0], result[1]


def _argmax_helper_pairwise(enc_tensor, dim=None):
    dim = -1 if dim is None else dim
    row_length = enc_tensor.size(dim) if enc_tensor.size(dim) > 1 else 2

    a = enc_tensor.expand(row_length - 1, *enc_tensor.size())
    b = prix.stack([enc_tensor.roll(i + 1, dims=dim) for i in range(row_length - 1)])

    if row_length - 1 < 64 * 2:
        pairwise_comparisons = a.ge(b)
        result = pairwise_comparisons.prod(0)
    else:
        pairwise_comparisons = a.ge(b)
        result = pairwise_comparisons.sum(0).ge(row_length - 1)
    return result, None


def _compute_pairwise_comparisons_for_steps(input_tensor, dim, steps):
    enc_tensor_reduced = input_tensor.clone()
    for _ in range(steps):
        m = enc_tensor_reduced.size(dim)
        if m <= 1:
            break
        x, y, remainder = enc_tensor_reduced.split([m // 2, m // 2, m % 2], dim=dim)
        pairwise_max = prix.where(x >= y, x, y)
        enc_tensor_reduced = prix.cat([pairwise_max, remainder], dim=dim)
    return enc_tensor_reduced


def _max_helper_log_reduction(enc_tensor, dim=None):
    if enc_tensor.dim() == 0:
        return enc_tensor
    inp, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        inp = enc_tensor.flatten()
    n = inp.size(dim_used)
    steps = int(math.log(builtins.max(n, 1)))
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(inp, dim_used, steps)
    with cfg.temp_override({"functions.max.method": "pairwise"}):
        enc_max_vec, _ = enc_tensor_reduced.max(dim=dim_used)
    return enc_max_vec


def _max_helper_double_log_recursive(enc_tensor, dim):
    n = enc_tensor.size(dim)
    sqrt_n = int(math.sqrt(n))
    count_sqrt_n = n // sqrt_n
    if n == 1:
        return enc_tensor
    else:
        size_arr = [sqrt_n * count_sqrt_n, n % sqrt_n]
        split_enc_tensor, remainder = enc_tensor.split(size_arr, dim=dim)

        updated_enc_tensor_size = [sqrt_n, enc_tensor.size(dim + 1) * count_sqrt_n]
        size_arr = [enc_tensor.size(i) for i in range(enc_tensor.dim())]
        size_arr[dim], size_arr[dim + 1] = updated_enc_tensor_size
        split_enc_tensor = split_enc_tensor.reshape(size_arr)

        split_enc_max = _max_helper_double_log_recursive(split_enc_tensor, dim)

        size_arr[dim], size_arr[dim + 1] = [count_sqrt_n, enc_tensor.size(dim + 1)]
        enc_max_tensor = split_enc_max.reshape(size_arr)
        full_max_tensor = prix.cat([enc_max_tensor, remainder], dim=dim)

        with cfg.temp_override({"functions.max.method": "pairwise"}):
            enc_max, _ = full_max_tensor.max(dim=dim, keepdim=True)
        return enc_max


def _max_helper_double_log_reduction(enc_tensor, dim=None):
    if enc_tensor.dim() == 0:
        return enc_tensor
    inp, dim_used, size_arr = enc_tensor, dim, ()
    if dim is None:
        dim_used = 0
        inp = enc_tensor.flatten()
    dim_used = dim_used + inp.dim() if dim_used < 0 else dim_used
    if inp.dim() > 1:
        size_arr = [inp.size(i) for i in range(inp.dim()) if i != dim_used]
    inp = inp.unsqueeze(dim_used + 1)
    enc_max_val = _max_helper_double_log_recursive(inp, dim_used)
    enc_max_val = enc_max_val.squeeze(dim_used + 1)
    enc_max_val = enc_max_val.reshape(size_arr)
    return enc_max_val


def _max_helper_accelerated_cascade(enc_tensor, dim=None):
    if enc_tensor.dim() == 0:
        return enc_tensor
    inp, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        inp = enc_tensor.flatten()
    n = inp.size(dim_used)
    if n < 3:
        with cfg.temp_override({"functions.max.method": "pairwise"}):
            enc_max, _ = inp.max(dim=dim_used)
            return enc_max
    steps = int(math.log(builtins.max(math.log(builtins.max(math.log(n), 1.01)), 1.01))) + 1
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(inp, dim_used, steps)
    return _max_helper_double_log_reduction(enc_tensor_reduced, dim=dim_used)


def _max_helper_all_tree_reductions(enc_tensor, dim=None, method="log_reduction"):
    if method == "log_reduction":
        return _max_helper_log_reduction(enc_tensor, dim)
    elif method == "double_log_reduction":
        return _max_helper_double_log_reduction(enc_tensor, dim)
    elif method == "accelerated_cascade":
        return _max_helper_accelerated_cascade(enc_tensor, dim)
    else:
        raise RuntimeError(f"Unknown max method: {method}")


def _argmax_helper_all_tree_reductions(enc_tensor, dim=None, method="log_reduction"):
    enc_max_vec = _max_helper_all_tree_reductions(enc_tensor, dim=dim, method=method)
    enc_max_vec_orig = enc_max_vec.unsqueeze(dim) if dim is not None else enc_max_vec
    enc_one_hot_vec = enc_tensor.eq(enc_max_vec_orig)
    return enc_one_hot_vec, enc_max_vec


def _argmax_helper(enc_tensor, dim=None, one_hot=True, method="pairwise", _return_max=False):
    if enc_tensor.dim() == 0:
        result = (
            enc_tensor.new(jnp.ones((), dtype=jnp.float32))
            if one_hot
            else enc_tensor.new(jnp.zeros((), dtype=jnp.float32))
        )
        if _return_max:
            return result, None
        return result

    updated_enc_tensor = enc_tensor.flatten() if dim is None else enc_tensor

    if method == "pairwise":
        result_args, result_val = _argmax_helper_pairwise(updated_enc_tensor, dim)
    elif method in ["log_reduction", "double_log_reduction", "accelerated_cascade"]:
        result_args, result_val = _argmax_helper_all_tree_reductions(
            updated_enc_tensor, dim, method
        )
    else:
        raise RuntimeError(f"Unknown argmax method: {method}")

    result_args = result_args.weighted_index(dim)
    result_args = result_args.view(enc_tensor.size()) if dim is None else result_args

    if _return_max:
        return result_args, result_val
    return result_args


def _one_hot_to_index(tensor, dim, keepdim, device=None):
    if dim is None:
        result = tensor.flatten()
        indices = jnp.arange(tensor.nelement(), dtype=jnp.int64)
        result = result * indices
        return result.sum()
    else:
        size = [1] * tensor.dim()
        size[dim] = tensor.size(dim)
        indices = jnp.arange(tensor.size(dim), dtype=jnp.int64).reshape(size)
        result = tensor * indices
        return result.sum(dim, keepdim=keepdim)


# Power / Polynomial / Norm functions
def pow(self, p, **kwargs):
    """Computes an element-wise integer power `p` of a tensor."""
    if isinstance(p, float) and int(p) == p:
        p = int(p)

    if not isinstance(p, int):
        raise TypeError(
            "pow must take an integer exponent. For non-integer powers, use"
            " pos_pow with positive-valued base."
        )
    if p < -1:
        return self.reciprocal().pow(-p)
    elif p == -1:
        return self.reciprocal()
    elif p == 0:
        return self.new(jnp.ones_like(self.share, dtype=jnp.float32))
    elif p == 1:
        return self.clone()
    elif p == 2:
        return self.square()
    elif p % 2 == 0:
        return self.square().pow(p // 2)
    else:
        x = self.square().mul(self)
        return x.pow((p - 1) // 2)


def pos_pow(self, p):
    """Approximates self ** p by computing exp(p * log(self))."""
    if isinstance(p, int) or (isinstance(p, float) and int(p) == p):
        return self.pow(int(p))
    return self.log().mul(p).exp()


def polynomial(self, coeffs, func="mul"):
    """Computes a polynomial function on a tensor with given coefficients."""
    if isinstance(coeffs, list):
        coeffs = jnp.asarray(coeffs, dtype=jnp.float32)
    assert isinstance(coeffs, (jnp.ndarray, np.ndarray, prix.Array)), (
        "Polynomial coefficients must be a list or tensor"
    )
    assert coeffs.ndim == 1, "Polynomial coefficients must be a 1-D tensor"

    if coeffs.shape[0] == 1:
        return self.mul(coeffs)

    terms = prix.stack([self, self.square()])
    while terms.size(0) < coeffs.shape[0]:
        highest_term = terms.index_select(0, jnp.asarray([terms.size(0) - 1]))
        new_terms = getattr(terms, func)(highest_term)
        terms = prix.cat([terms, new_terms])

    terms = terms[: coeffs.shape[0]]
    for _ in range(terms.dim() - 1):
        coeffs = jnp.expand_dims(coeffs, 1) if not isinstance(coeffs, prix.Array) else coeffs.unsqueeze(1)

    return terms.mul(coeffs).sum(0)


def norm(self, p="fro", dim=None, keepdim=False):
    """Computes the p-norm of the input tensor."""
    if p == "fro":
        p = 2

    if isinstance(p, (int, float)):
        assert p >= 1, "p-norm requires p >= 1"
        if p == 1:
            if dim is None:
                return self.abs().sum()
            return self.abs().sum(dim, keepdim=keepdim)
        elif p == 2:
            if dim is None:
                return self.square().sum().sqrt()
            return self.square().sum(dim, keepdim=keepdim).sqrt()
        elif p == float("inf"):
            if dim is None:
                return self.abs().max()
            return self.abs().max(dim=dim, keepdim=keepdim)[0]
        else:
            if dim is None:
                return self.abs().pos_pow(p).sum().pos_pow(1 / p)
            return self.abs().pos_pow(p).sum(dim, keepdim=keepdim).pos_pow(1 / p)
    elif p == "nuc":
        raise NotImplementedError("Nuclear norm is not implemented")
    else:
        raise ValueError(f"Improper value p ({p}) for p-norm")


# Dropout & Sampling
def dropout(self, p=0.5, training=True, inplace=False):
    """Randomly zeroes some elements of the input tensor with probability `p`."""
    if p == 0.0:
        return self if inplace else self.clone()
    elif p == 1.0:
        return self - self

    assert 0.0 < p < 1.0, "dropout probability has to be between 0 and 1"
    if not training:
        return self if inplace else self.clone()

    rand_tensor = prix.rand(self.size(), device=self.device)
    dropout_tensor = rand_tensor > p
    if inplace:
        result_tensor = self.div_(1 - p)
        return result_tensor.mul_(dropout_tensor)
    else:
        result_tensor = self.div(1 - p)
        return result_tensor.mul(dropout_tensor)


def weighted_index(self, dim=None):
    """Returns a tensor with entries that are one-hot along dimension `dim`."""
    if dim is None:
        return self.flatten().weighted_index(dim=0).view(self.size())

    dim_pos = dim + self.dim() if dim < 0 else dim
    x = self.cumsum(dim_pos)
    max_weight = x.index_select(dim_pos, jnp.asarray([x.size(dim_pos) - 1]))
    r = prix.rand(max_weight.size(), device=self.device) * max_weight

    gt_val = x.gt(r)
    shifted = gt_val.roll(1, dims=dim_pos)
    # Zero out index 0 along dim_pos
    idx = [slice(None)] * shifted.dim()
    idx[dim_pos] = 0
    shifted.share = shifted.share.at[tuple(idx)].set(0)

    return gt_val - shifted


def weighted_sample(self, dim=None):
    """Samples a single value across dimension `dim` with weights in `self`."""
    indices = self.weighted_index(dim)
    sample = self.mul(indices).sum(dim)
    return sample, indices


def bernoulli(self):
    """Returns a tensor with elements in {0, 1} sampled with probability `self`."""
    return self > prix.rand(self.size(), device=self.device)


def randn(*sizes, device=None):
    """Returns an encrypted Array with normally distributed elements via Box-Muller."""
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    u = prix.rand(*sizes, device=device).flatten()
    odd_numel = u.numel() % 2 == 1
    if odd_numel:
        u = prix.cat([u, prix.rand((1,), device=device)])

    n = u.numel() // 2
    u1 = u[:n]
    u2 = u[n:]

    r2 = -2 * u1.log(input_in_01=True)
    r = r2.sqrt()
    cos_val, sin_val = u2.sub(0.5).mul(6.28318531).cossin()

    x = r.mul(sin_val)
    y = r.mul(cos_val)
    z = prix.cat([x, y])

    if odd_numel:
        z = z[1:]

    return z.view(*sizes)


# 2D Pooling
def _pooling_output_shape(input_size, kernel_size, pad_l, pad_r, stride, dilation, ceil_mode):
    numerator = input_size + pad_l + pad_r - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        numerator += stride - 1
    output_size = numerator // stride + 1
    if ceil_mode and (output_size - 1) * stride >= input_size + pad_l:
        output_size -= 1
    return output_size


def _torch_pad_2d(arr, padding_4d, pad_value=0):
    pad_l, pad_r, pad_t, pad_b = padding_4d
    pad_widths = [(0, 0)] * (arr.ndim - 2) + [(pad_t, pad_b), (pad_l, pad_r)]
    return jnp.pad(arr, pad_widths, mode="constant", constant_values=pad_value)


def _pool2d_reshape(
    inp,
    kernel_size,
    padding=None,
    stride=None,
    dilation=1,
    ceil_mode=False,
    pad_value=0,
):
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    s = k if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
    d = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    if padding is not None:
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        padding_4d = (padding[1], padding[1], padding[0], padding[0])
    else:
        padding_4d = (0, 0, 0, 0)

    n = inp.shape[:-2]
    h = _pooling_output_shape(
        inp.shape[-2], k[0], padding_4d[2], padding_4d[3], s[0], d[0], ceil_mode
    )
    w = _pooling_output_shape(
        inp.shape[-1], k[1], padding_4d[0], padding_4d[1], s[1], d[1], ceil_mode
    )
    out_size = tuple(n + (h, w))

    pad_val = pad_value if comm.get().get_rank() == 0 else 0
    inp = _torch_pad_2d(inp, padding_4d, pad_value=pad_val)
    if ceil_mode:
        update_pad = [0, 0, 0, 0]
        update_pad[3] = builtins.max(0, h * s[0] + (k[0] - 1) * d[0] - inp.shape[-2])
        update_pad[1] = builtins.max(0, w * s[1] + (k[1] - 1) * d[1] - inp.shape[-1])
        inp = _torch_pad_2d(inp, tuple(update_pad), pad_value=pad_val)

    kernel_indices = jnp.arange(0, k[1] * d[1], d[1], dtype=jnp.int64)
    kernel_indices = jnp.concatenate(
        [kernel_indices + i * inp.shape[-1] for i in range(0, k[0] * d[0], d[0])]
    )
    kernel_indices = jnp.stack([kernel_indices + i * s[1] for i in range(w)])

    offset = inp.shape[-1]
    kernel_indices = jnp.concatenate([kernel_indices + i * s[0] * offset for i in range(h)])

    for dim in range(2, inp.ndim):
        offset *= inp.shape[-dim]
        kernel_indices = jnp.stack(
            [kernel_indices + i * offset for i in range(inp.shape[-dim - 1])]
        )

    output = jnp.take(inp, kernel_indices)
    return output, out_size


def max_pool2d(
    self,
    kernel_size,
    padding=0,
    stride=None,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies a 2D max pooling over an input signal."""
    max_input = self.clone()
    max_input.share, output_size = _pool2d_reshape(
        self.share,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        ceil_mode=ceil_mode,
        pad_value=-(2**24),
    )
    max_vals, argmax_vals = max_input.max(dim=-1, one_hot=True)
    max_vals = max_vals.view(output_size)
    if return_indices:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        argmax_vals = argmax_vals.view(output_size + tuple(kernel_size))
        return max_vals, argmax_vals
    return max_vals


def adaptive_avg_pool2d(self, output_size):
    """Applies a 2D adaptive average pooling over an input signal."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if output_size is None or output_size[0] is None:
        output_size = self.shape[-2:]
    if self.shape[-2:] == tuple(output_size):
        return self.clone()

    in_h, in_w = self.shape[-2], self.shape[-1]
    out_h, out_w = output_size
    stride_h = in_h // out_h
    stride_w = in_w // out_w
    kernel_h = in_h - (out_h - 1) * stride_h
    kernel_w = in_w - (out_w - 1) * stride_w
    return self.avg_pool2d((kernel_h, kernel_w), stride=(stride_h, stride_w))


def adaptive_max_pool2d(self, output_size, return_indices=False):
    """Applies a 2D adaptive max pooling over an input signal."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if output_size is None or output_size[0] is None:
        output_size = self.shape[-2:]
    if self.shape[-2:] == tuple(output_size):
        if return_indices:
            return self.clone(), self.new(jnp.ones(self.shape + tuple(output_size)))
        return self.clone()

    in_h, in_w = self.shape[-2], self.shape[-1]
    out_h, out_w = output_size
    stride_h = in_h // out_h
    stride_w = in_w // out_w
    kernel_h = in_h - (out_h - 1) * stride_h
    kernel_w = in_w - (out_w - 1) * stride_w
    return self.max_pool2d(
        (kernel_h, kernel_w), stride=(stride_h, stride_w), return_indices=return_indices
    )
