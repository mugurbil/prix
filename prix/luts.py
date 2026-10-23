#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wavelet look-up tables (Haar / bior(5,3)) for PriX's nonlinearities.

`LookupTables` builds, caches (in memory and on disk) and serves the fixed-point tables that
the LUT-based modes (`curl`, `wave`, `sigma`, `default`) evaluate with FSS or one-hot protocols.
Tables are regenerated lazily from the current `prix.cfg` whenever the configured domain
(`lut_max_bits`), resolution (`*_size_bits`) or `precision_bits` change.
"""

import builtins
import hashlib
import logging
import math
import os
import stat
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pywt

try:
    from scipy.special import erf as _scipy_erf
except ImportError:
    _scipy_erf = None

from prix.config import cfg
from prix.encoder import DATATYPE, resolve_device, to_jax

__all__ = ["LookupTables"]


class _LazyLUTDict(dict):
    def __getitem__(self, key):
        if LookupTables._initialized_precision != cfg.encoder.precision_bits:
            self.clear()
            LookupTables._initialized_precision = cfg.encoder.precision_bits
        if key in LookupTables._BUILTIN_LUT_NAMES or not super().__contains__(key):
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
    _mem_cache = {}
    # Cap on log2(#samples) used to build a bior table (see `generate_bior`).
    _MAX_BIOR_SAMPLE_BITS = 24
    _initialized_precision = None
    _default_device = None
    _BUILTIN_LUT_NAMES = frozenset({
        "exp_haar", "exp_bior", "nexp_low", "nexp_high", "nexp_haar", "nexp_bior",
        "log_haar", "log_bior", "reciprocal_haar", "reciprocal_bior",
        "sqrt_haar", "sqrt_bior", "inv_sqrt_haar", "inv_sqrt_bior",
        "inv_sqrt_tailored_haar_0", "inv_sqrt_tailored_haar_1",
        "sin_haar", "sin_bior", "sin_haar_lut_only", "sin_bior_lut_only",
        "cos_haar", "cos_bior", "cos_haar_lut_only", "cos_bior_lut_only",
        "sigmoid_haar", "sigmoid_bior", "sigmoid_haar_lut_only", "sigmoid_bior_lut_only",
        "tanh_haar", "tanh_bior", "tanh_haar_lut_only", "tanh_bior_lut_only",
        "erf_haar", "erf_bior", "erf_haar_lut_only", "erf_bior_lut_only",
        "gelu_haar", "gelu_bior", "gelu_haar_lut_only", "gelu_bior_lut_only",
        "silu_haar", "silu_bior", "silu_haar_lut_only", "silu_bior_lut_only",
    })

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

    _LUT_CACHE_VERSION = "v2"

    @classmethod
    def _safe_cache_dir(cls):
        uid = os.getuid() if hasattr(os, "getuid") else "default"
        cache_dir = os.path.join(tempfile.gettempdir(), f"prix_lut_cache_{uid}")
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
        st = os.lstat(cache_dir)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise RuntimeError(f"Refusing to use non-directory LUT cache path: {cache_dir}")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise RuntimeError(f"LUT cache directory {cache_dir} is not owned by current user")
        if hasattr(os, "chmod") and (st.st_mode & 0o077) != 0:
            os.chmod(cache_dir, 0o700)
        return cache_dir

    @classmethod
    def _cache_path(cls, kind, name, max_bits, lut_bits, negative_values, function=None):
        cache_dir = cls._safe_cache_dir()
        code = getattr(function, "__code__", None)
        if code is not None:
            func_bytes = code.co_code + repr(code.co_consts).encode("utf-8")
        else:
            func_bytes = getattr(function, "__name__", repr(function)).encode("utf-8")
        fn_hash = hashlib.sha256(func_bytes).hexdigest()[:12]
        fname = (
            f"{cls._LUT_CACHE_VERSION}_{kind}_{name}_p{cfg.encoder.precision_bits}"
            f"_m{max_bits}_l{lut_bits}_n{int(bool(negative_values))}_{fn_hash}.npy"
        )
        return os.path.join(cache_dir, fname)

    @classmethod
    def _load_cached_lut(cls, cpath, expected_shape):
        if not os.path.exists(cpath):
            return None
        try:
            st = os.lstat(cpath)
            if not stat.S_ISREG(st.st_mode):
                return None
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                return None
            arr_np = np.load(cpath, allow_pickle=False)
            if arr_np.dtype != np.int64 or arr_np.shape != expected_shape:
                return None
            return arr_np
        except Exception:
            return None

    @classmethod
    def _save_cached_lut(cls, cpath, arr_np):
        try:
            tmp_p = f"{cpath}.{os.getpid()}.tmp.npy"
            np.save(tmp_p, arr_np, allow_pickle=False)
            if hasattr(os, "chmod"):
                os.chmod(tmp_p, 0o600)
            os.replace(tmp_p, cpath)
        except Exception:
            pass

    @classmethod
    def generate_haar(cls, max_bits, lut_bits, function, name, negative_values=False, device=None):
        mem_key = (
            "haar",
            name,
            int(cfg.encoder.precision_bits),
            int(max_bits),
            int(lut_bits),
            bool(negative_values),
            str(device),
        )
        if mem_key in cls._mem_cache:
            dict.__setitem__(cls.LUTs, name, cls._mem_cache[mem_key])
            return
        expected_shape = (2 ** (int(lut_bits) + int(bool(negative_values))),)
        cpath = None
        try:
            cpath = cls._cache_path("haar", name, max_bits, lut_bits, negative_values, function=function)
            cached = cls._load_cached_lut(cpath, expected_shape)
            if cached is not None:
                arr_i64 = jnp.asarray(cached, dtype=DATATYPE)
                if device is not None:
                    arr_i64 = jax.device_put(arr_i64, resolve_device(device))
                cls._mem_cache[mem_key] = arr_i64
                dict.__setitem__(cls.LUTs, name, arr_i64)
                return
        except Exception:
            cpath = None
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
        if cpath is not None:
            cls._save_cached_lut(cpath, arr_np)
        arr_i64 = jnp.asarray(arr_np, dtype=DATATYPE)
        if device is not None:
            arr_i64 = jax.device_put(arr_i64, resolve_device(device))
        cls._mem_cache[mem_key] = arr_i64
        dict.__setitem__(cls.LUTs, name, arr_i64)

    @classmethod
    def generate_bior(cls, max_bits, lut_bits, function, name, negative_values=False, device=None):
        mem_key = (
            "bior",
            name,
            int(cfg.encoder.precision_bits),
            int(max_bits),
            int(lut_bits),
            bool(negative_values),
            str(device),
        )
        if mem_key in cls._mem_cache:
            dict.__setitem__(cls.LUTs, name, cls._mem_cache[mem_key])
            return
        expected_shape = (2, 2 ** (int(lut_bits) + int(bool(negative_values))))
        cpath = None
        try:
            cpath = cls._cache_path("bior", name, max_bits, lut_bits, negative_values, function=function)
            cached = cls._load_cached_lut(cpath, expected_shape)
            if cached is not None:
                arr_i64 = jnp.asarray(cached, dtype=DATATYPE)
                if device is not None:
                    arr_i64 = jax.device_put(arr_i64, resolve_device(device))
                cls._mem_cache[mem_key] = arr_i64
                dict.__setitem__(cls.LUTs, name, arr_i64)
                return
        except Exception:
            cpath = None
        scale = 2 ** cfg.encoder.precision_bits
        max_element = 2 ** max_bits
        depth = max_bits + cfg.encoder.precision_bits - lut_bits
        # The bior2.2 approximation coefficients at level `depth` are local weighted averages of
        # the function around each knot; sampling the function at 2^sample_bits points per unit
        # instead of 2^f changes them only by the (negligible) smoothing error but lets us build
        # tables over wide domains (e.g. [0, 2^16) for the range-extended inv_sqrt) without
        # materialising 2^(max_bits + f) samples. The wavelet normalisation is then rescaled from
        # 2^(depth_eff/2) to the 2^(depth/2) that the on-line protocol expects.
        sample_bits = builtins.min(int(cfg.encoder.precision_bits), builtins.max(lut_bits - max_bits + 2, cls._MAX_BIOR_SAMPLE_BITS - max_bits))
        sample_scale = 2 ** sample_bits
        depth_eff = max_bits + sample_bits - lut_bits
        if negative_values:
            xs = np.linspace(-max_element + 1.0 / sample_scale, max_element, int(2 * max_element * sample_scale))
            full = function(xs)
            coeffs, *_ = pywt.wavedec(full, "bior2.2", level=depth_eff)
            coeffs = np.stack(
                [
                    np.roll(coeffs, -2)[: 2 ** (lut_bits + 1)],
                    np.roll(coeffs, -3)[: 2 ** (lut_bits + 1)],
                ]
            )
        else:
            xs = np.linspace(1.0 / sample_scale, max_element, int(max_element * sample_scale))
            full = function(xs)
            coeffs, *_ = pywt.wavedec(full, "bior2.2", level=depth_eff)
            coeffs = np.stack(
                [
                    np.roll(coeffs, -2)[: 2 ** lut_bits],
                    np.roll(coeffs, -3)[: 2 ** lut_bits],
                ]
            )
        arr_np = cls._to_int64_np((coeffs * scale) * (2 ** (depth - 0.5 * depth_eff)))
        if cpath is not None:
            cls._save_cached_lut(cpath, arr_np)
        arr_i64 = jnp.asarray(arr_np, dtype=DATATYPE)
        if device is not None:
            arr_i64 = jax.device_put(arr_i64, resolve_device(device))
        cls._mem_cache[mem_key] = arr_i64
        dict.__setitem__(cls.LUTs, name, arr_i64)

    @classmethod
    def get_lut(cls, name, device=None):
        if cls._initialized_precision != cfg.encoder.precision_bits:
            cls.LUTs.clear()
            cls._initialized_precision = cfg.encoder.precision_bits
        if name in cls._BUILTIN_LUT_NAMES or not dict.__contains__(cls.LUTs, name):
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
            size = int(cfg.functions.exp.neg_lut_size)
            mem_key = ("split_nexp", int(cfg.encoder.precision_bits), size, str(device))
            if mem_key not in cls._mem_cache:
                idx = np.arange(size, dtype=np.float64)
                full_low = np.exp(-idx * (16.0 / (size * size)))
                full_high = np.exp(-idx * (16.0 / size))
                cls._mem_cache[mem_key] = (
                    cls._to_jax_lut(full_low * scale, device=device),
                    cls._to_jax_lut(full_high * scale, device=device),
                )
            low_lut, high_lut = cls._mem_cache[mem_key]
            dict.__setitem__(cls.LUTs, "nexp_low", low_lut)
            dict.__setitem__(cls.LUTs, "nexp_high", high_lut)
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
        elif name.startswith("inv_sqrt_bior_s"):
            # Window table for the range-extended inv_sqrt: T_e(v) = 2^(e/2) * v^(-1/2), so that
            # x^(-1/2) = T_e(2^e x). Folding 2^(e/2) into the table keeps full output precision.
            e = int(name[len("inv_sqrt_bior_s"):])
            cls.generate_bior(
                cfg.functions.inv_sqrt.lut_max_bits,
                cfg.functions.inv_sqrt.bior_size_bits,
                lambda x, _k=float(2 ** (e / 2)): _k * np.reciprocal(np.sqrt(x)),
                name,
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
