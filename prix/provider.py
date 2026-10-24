#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

import prix.communicator as comm
from prix.communicator import wire
from prix.config import cfg
from prix.encoder import DATATYPE, resolve_device, to_jax
from prix.random import (
    PRNGGenerator,
    _sm64_raw,
    count_wraps,
    generate_kbit_random_tensor,
    generate_kbit_random_tensor_np,
    generate_permutation,
    generate_random_ring_element,
    generate_random_ring_element_np,
)


# Largest contraction size K for which each limb-product GEMM is exact in float64
# (every entry of A_i @ B_j must stay below 2^53):
#   22-bit limbs: K * 2^44 <= 2^53  ->  K <= 2^9
#   16-bit limbs: K * 2^32 <= 2^53  ->  K <= 2^21
# Larger contractions are split into chunks of at most _MATMUL_MAX_K_16 and the
# exact uint64 partial results are summed (wrap-around addition is exact mod 2^64).
_MATMUL_MAX_K_22 = 1 << 9
_MATMUL_MAX_K_16 = 1 << 21


# ==============================================================================
# Preprocessing (dealer key) size accounting.
#
# PriX's TFP provider is a *simulated* dealer: both parties expand the same PRG seeds
# locally, so no key material is ever transmitted. To report what a real dealer would
# have to send to each party, every TFP generation routine calls `_acct_preproc` with
# the number of bits of non-seed-derivable material in the key of ONE party:
#   - FSS correction words (received identically by both parties): one 64-bit word per
#     DPF level in PriX (62-bit seed + 2 control bits). `bits_l128` re-counts each seed
#     CW at lambda = 128 (16 B / level, as in SIGMA / Orca) for a like-for-like comparison.
#   - Shared ring elements (Beaver triples, masks, B2A bits): with pairwise dealer<->party
#     seeds only ONE party receives each share explicitly; we charge the full 64 bits
#     (1 bit for binary masks) per shared value, i.e. the worst-case party.
# ==============================================================================
_PREPROC_STATS = {"bits": 0, "bits_l128": 0, "calls": 0}


def reset_preproc_stats():
    _PREPROC_STATS["bits"] = 0
    _PREPROC_STATS["bits_l128"] = 0
    _PREPROC_STATS["calls"] = 0


def get_preproc_stats():
    """Returns `{"bytes": ..., "bytes_l128": ..., "calls": ...}` accumulated since the last reset."""
    return {
        "bytes": _PREPROC_STATS["bits"] / 8.0,
        "bytes_l128": _PREPROC_STATS["bits_l128"] / 8.0,
        "calls": _PREPROC_STATS["calls"],
    }


def _acct_preproc(bits, bits_l128=None):
    _PREPROC_STATS["bits"] += int(bits)
    _PREPROC_STATS["bits_l128"] += int(bits if bits_l128 is None else bits_l128)
    _PREPROC_STATS["calls"] += 1


def _acct_fss_key(n, dpf_levels=0, dcf_levels=0, ring_elems=0, bit_elems=0):
    """
    Accounts one party's FSS key for `n` elements:
      - DPF: `dpf_levels` seed correction words (64 bits each in PriX, 128 at lambda=128).
      - Arithmetic DCF: `dcf_levels` seed CWs + `dcf_levels` 64-bit value CWs + 1 final 64-bit CW.
      - `ring_elems` 64-bit shares and `bit_elems` 1-bit shares.
    """
    seed_bits = dpf_levels + dcf_levels
    value_bits = (dcf_levels + (1 if dcf_levels > 0 else 0)) * 64
    share_bits = ring_elems * 64 + bit_elems
    bits = n * (seed_bits * 64 + value_bits + share_bits)
    bits_l128 = n * (seed_bits * 128 + value_bits + share_bits)
    _acct_preproc(bits, bits_l128)


@jax.jit
def _int64_matmul(a, b):
    """
    Bit-exact 64-bit ring matrix multiplication mod 2^64 via limb splitting in float64:
    - For K <= 512: 3 limbs of 22 bits (6 float64 matmuls instead of CrypTen's 9).
    - For K >  512: 4 limbs of 16 bits (10 float64 matmuls instead of CrypTen's 16).
    - For K > 2^21: the contraction is split into chunks of at most 2^21, each computed
      exactly as above, and the partial results are summed mod 2^64.
    On GPU, limbs are stacked along axis 0 to execute a single cublasDgemmStridedBatched kernel.
    """
    k = a.shape[-1]
    if k <= _MATMUL_MAX_K_16:
        return _int64_matmul_exact(a, b)
    b_axis = b.ndim - 2 if b.ndim >= 2 else 0
    res = None
    for start in range(0, k, _MATMUL_MAX_K_16):
        stop = min(start + _MATMUL_MAX_K_16, k)
        part = _int64_matmul_exact(
            jax.lax.slice_in_dim(a, start, stop, axis=a.ndim - 1),
            jax.lax.slice_in_dim(b, start, stop, axis=b_axis),
        )
        res = part if res is None else res + part  # int64 addition wraps mod 2^64
    return res


def _int64_matmul_exact(a, b):
    """Exact limb-split matmul for a single chunk with K <= _MATMUL_MAX_K_16 (traced inside jit)."""
    ua = jax.lax.bitcast_convert_type(a, jnp.uint64)
    ub = jax.lax.bitcast_convert_type(b, jnp.uint64)
    k = a.shape[-1]
    is_gpu = jax.default_backend() == "gpu"
    if is_gpu:
        if ua.ndim > ub.ndim:
            ub = ub.reshape((1,) * (ua.ndim - ub.ndim) + ub.shape)
        elif ub.ndim > ua.ndim:
            ua = ua.reshape((1,) * (ub.ndim - ua.ndim) + ua.shape)
    if k <= _MATMUL_MAX_K_22:
        mask22 = jnp.uint64((1 << 22) - 1)
        a0 = (ua & mask22).astype(jnp.float64)
        a1 = ((ua >> 22) & mask22).astype(jnp.float64)
        a2 = (ua >> 44).astype(jnp.float64)
        b0 = (ub & mask22).astype(jnp.float64)
        b1 = ((ub >> 22) & mask22).astype(jnp.float64)
        b2 = (ub >> 44).astype(jnp.float64)

        if is_gpu:
            lhs = jnp.stack([a0, a0, a1, a0, a1, a2], axis=0)
            rhs = jnp.stack([b0, b1, b0, b2, b1, b0], axis=0)
            p = jnp.matmul(lhs, rhs).astype(jnp.uint64)
            c0 = p[0]
            c1 = p[1] + p[2]
            c2 = p[3] + p[4] + p[5]
        elif k <= 128:
            c0 = (a0 @ b0).astype(jnp.uint64)
            c1 = (a0 @ b1 + a1 @ b0).astype(jnp.float64).astype(jnp.uint64)
            c2 = (a0 @ b2 + a1 @ b1 + a2 @ b0).astype(jnp.float64).astype(jnp.uint64)
        else:
            c0 = (a0 @ b0).astype(jnp.uint64)
            c1 = (a0 @ b1).astype(jnp.uint64) + (a1 @ b0).astype(jnp.uint64)
            c2 = (
                (a0 @ b2).astype(jnp.uint64)
                + (a1 @ b1).astype(jnp.uint64)
                + (a2 @ b0).astype(jnp.uint64)
            )
        res = c0 + (c1 << jnp.uint64(22)) + (c2 << jnp.uint64(44))
        return jax.lax.bitcast_convert_type(res, DATATYPE)
    else:
        mask16 = jnp.uint64(0xFFFF)
        a0 = (ua & mask16).astype(jnp.float64)
        a1 = ((ua >> 16) & mask16).astype(jnp.float64)
        a2 = ((ua >> 32) & mask16).astype(jnp.float64)
        a3 = (ua >> 48).astype(jnp.float64)
        b0 = (ub & mask16).astype(jnp.float64)
        b1 = ((ub >> 16) & mask16).astype(jnp.float64)
        b2 = ((ub >> 32) & mask16).astype(jnp.float64)
        b3 = (ub >> 48).astype(jnp.float64)

        if is_gpu:
            lhs = jnp.stack([a0, a0, a1, a0, a1, a2, a0, a1, a2, a3], axis=0)
            rhs = jnp.stack([b0, b1, b0, b2, b1, b0, b3, b2, b1, b0], axis=0)
            p = jnp.matmul(lhs, rhs).astype(jnp.uint64)
            c0 = p[0]
            c1 = p[1] + p[2]
            c2 = p[3] + p[4] + p[5]
            c3 = p[6] + p[7] + p[8] + p[9]
        else:
            # Convert each limb product to uint64 before summing, so every float64
            # value stays below K * 2^32 <= 2^53 (exact for K <= 2^21, as on GPU).
            u = lambda x: x.astype(jnp.uint64)
            c0 = u(a0 @ b0)
            c1 = u(a0 @ b1) + u(a1 @ b0)
            c2 = u(a0 @ b2) + u(a1 @ b1) + u(a2 @ b0)
            c3 = u(a0 @ b3) + u(a1 @ b2) + u(a2 @ b1) + u(a3 @ b0)
        res = c0 + (c1 << jnp.uint64(16)) + (c2 << jnp.uint64(32)) + (c3 << jnp.uint64(48))
        return jax.lax.bitcast_convert_type(res, DATATYPE)


def _eval_ring_op(op, a, b, *args, **kwargs):
    """Evaluate a ring operation ('mul', 'matmul', 'conv1d', 'conv2d') on int64 JAX arrays."""
    if op == "mul":
        return a * b
    elif op == "matmul":
        return _int64_matmul(a, b)
    elif op == "conv1d":
        stride = kwargs.get("stride", args[0] if len(args) > 0 else 1)
        padding = kwargs.get("padding", args[1] if len(args) > 1 else 0)
        dilation = kwargs.get("dilation", args[2] if len(args) > 2 else 1)
        groups = kwargs.get("groups", args[3] if len(args) > 3 else 1)
        s = (stride,) if isinstance(stride, int) else tuple(stride)
        p = [(padding, padding)] if isinstance(padding, int) else [(padding[0], padding[0])]
        d = (dilation,) if isinstance(dilation, int) else tuple(dilation)
        return jax.lax.conv_general_dilated(
            a,
            b,
            window_strides=s,
            padding=p,
            rhs_dilation=d,
            feature_group_count=groups,
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
    elif op == "conv2d":
        stride = kwargs.get("stride", args[0] if len(args) > 0 else 1)
        padding = kwargs.get("padding", args[1] if len(args) > 1 else 0)
        dilation = kwargs.get("dilation", args[2] if len(args) > 2 else 1)
        groups = kwargs.get("groups", args[3] if len(args) > 3 else 1)
        s = (stride, stride) if isinstance(stride, int) else tuple(stride)
        if isinstance(padding, int):
            p = [(padding, padding), (padding, padding)]
        else:
            p = [(padding[0], padding[0]), (padding[1], padding[1])]
        d = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        return jax.lax.conv_general_dilated(
            a,
            b,
            window_strides=s,
            padding=p,
            rhs_dilation=d,
            feature_group_count=groups,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
    else:
        raise ValueError(f"Unsupported ring operation: {op}")


class TupleProvider:
    """Base class for Beaver tuple and correlated randomness providers."""

    TRACEABLE_FUNCTIONS = [
        "generate_additive_triple",
        "generate_linear_triple",
        "generate_dcf_key",
        "generate_wave_key",
        "square",
        "generate_binary_triple",
        "wrap_rng",
        "B2A_rng",
        "generate_one_hot",
        "egk_trunc_pr_rng",
        "generate_trig_triple",
    ]

    _DEFAULT_CACHE_PATH = os.path.join(
        tempfile.gettempdir(),
        f"prix_tuple_cache_{os.getuid() if hasattr(os, 'getuid') else 'default'}",
    )
    CACHE_SAVE_BATCH_SIZE = 1000

    def __init__(self, device=None):
        self.tracing = False
        self.request_cache = []
        self.tuple_cache = {}
        self.device = device

    @property
    def rank(self):
        return comm.get().get_rank()

    def _get_request_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
        os.makedirs(prefix, mode=0o700, exist_ok=True)
        return os.path.join(prefix, f"request_cache-{self.rank}.bin")

    def _get_tuple_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
        os.makedirs(prefix, mode=0o700, exist_ok=True)
        return os.path.join(prefix, f"tuple_cache-{self.rank}")

    def trace(self, tracing=True):
        self.tracing = tracing

    def trace_once(self):
        untraced = len(self.request_cache) == 0
        self.trace(tracing=untraced)

    def _save_requests(self, filepath=None):
        if len(self.request_cache) == 0:
            return
        path = self._get_request_path(prefix=filepath)
        serialized = [_serialize_cache_obj(req) for req in self.request_cache]
        with open(path, "wb") as f:
            for part in wire.encode(serialized):
                f.write(part)
        self.request_cache = []

    def _load_requests(self, filepath=None):
        path = self._get_request_path(prefix=filepath)
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw = wire.decode(bytearray(f.read()))
            self.request_cache = [_deserialize_cache_obj(req, device=self.device) for req in raw]

    def _save_tuples(self, filepath=None):
        if len(self.tuple_cache) == 0:
            return
        dirpath = self._get_tuple_path(prefix=filepath)
        os.makedirs(dirpath, mode=0o700, exist_ok=True)
        existing_files = [
            f for f in os.listdir(dirpath) if f.startswith("tuple_batch_")
        ]
        next_index = len(existing_files)
        batch_file = os.path.join(dirpath, f"tuple_batch_{next_index}.bin")
        tensor_cache = {k: [_serialize_cache_obj(item) for item in v] for k, v in self.tuple_cache.items()}
        with open(batch_file, "wb") as f:
            for part in wire.encode(tensor_cache):
                f.write(part)
        self.tuple_cache.clear()

    def _load_tuples(self, filepath=None):
        dirpath = self._get_tuple_path(prefix=filepath)
        if not os.path.exists(dirpath):
            return
        batch_files = sorted(
            f for f in os.listdir(dirpath) if f.startswith("tuple_batch_")
        )
        self.tuple_cache = {}
        for batch_file in batch_files:
            batch_path = os.path.join(dirpath, batch_file)
            with open(batch_path, "rb") as f:
                batch_data = wire.decode(bytearray(f.read()))
            for key, values in batch_data.items():
                deser = [_deserialize_cache_obj(item, device=self.device) for item in values]
                self.tuple_cache.setdefault(key, []).extend(deser)

    def load_cache(self, filepath=None):
        self._load_requests(filepath=filepath)
        self._load_tuples(filepath=filepath)

    def remove_cache(self):
        req_path = self._get_request_path()
        if os.path.exists(req_path):
            os.remove(req_path)
        dirpath = self._get_tuple_path()
        if os.path.exists(dirpath):
            for f in os.listdir(dirpath):
                if f.startswith("tuple_batch_"):
                    os.remove(os.path.join(dirpath, f))

    def fill_cache(self):
        self.remove_cache()
        batch_count = 0
        for func_name, args, kwargs in self.request_cache:
            result = object.__getattribute__(self, func_name)(*args, **kwargs)
            hashable_request = _cache_key(func_name, args, kwargs)
            # Each request gets its own fresh tuple: correlated randomness must
            # never be reused (opening x - a and x' - a would reveal x - x').
            self.tuple_cache.setdefault(hashable_request, []).append(result)
            batch_count += 1
            if batch_count >= self.CACHE_SAVE_BATCH_SIZE:
                self._save_tuples()
                batch_count = 0
        if len(self.tuple_cache) > 0:
            self._save_tuples()
        self._save_requests()

    def __getattribute__(self, func_name):
        if func_name not in TupleProvider.TRACEABLE_FUNCTIONS:
            return object.__getattribute__(self, func_name)

        if object.__getattribute__(self, "tracing"):
            def func_with_trace(*args, **kwargs):
                object.__getattribute__(self, "request_cache").append(
                    (func_name, args, kwargs)
                )
                return object.__getattribute__(self, func_name)(*args, **kwargs)

            return func_with_trace

        tuple_cache = object.__getattribute__(self, "tuple_cache")
        if len(tuple_cache) == 0:
            return object.__getattribute__(self, func_name)

        def func_from_cache(*args, **kwargs):
            request = _cache_key(func_name, args, kwargs)
            pool = tuple_cache.get(request)
            if pool:
                # Single-use: pop so the same tuple is never handed out twice.
                result = pool.pop(0)
                if not pool:
                    del tuple_cache[request]
                return result
            return object.__getattribute__(self, func_name)(*args, **kwargs)

        return func_from_cache


def _serialize_cache_obj(v):
    from prix.array import Array

    if isinstance(v, Array):
        return ("__prix_array__", np.asarray(v.share), v.precision, str(v.ptype.value))
    if isinstance(v, tuple):
        return tuple(_serialize_cache_obj(x) for x in v)
    if isinstance(v, list):
        return [_serialize_cache_obj(x) for x in v]
    if isinstance(v, dict):
        return {k: _serialize_cache_obj(x) for k, x in v.items()}
    return v


def _deserialize_cache_obj(v, device=None):
    from prix.array import Array, Ptype, SENTINEL

    if isinstance(v, tuple) and len(v) == 4 and v[0] == "__prix_array__":
        _, share_np, precision, ptype_val = v
        arr = Array(src=SENTINEL, precision=precision, device=device, ptype=Ptype(ptype_val))
        arr.share = to_jax(share_np, device=device, dtype=DATATYPE)
        return arr
    if isinstance(v, tuple):
        return tuple(_deserialize_cache_obj(x, device=device) for x in v)
    if isinstance(v, list):
        return [_deserialize_cache_obj(x, device=device) for x in v]
    if isinstance(v, dict):
        return {k: _deserialize_cache_obj(x, device=device) for k, x in v.items()}
    return v


def _cache_key(func_name, args, kwargs):
    """Hashable tuple-cache key that includes kwargs (e.g. conv stride/padding)."""
    def _freeze(v):
        if isinstance(v, (list, tuple)):
            return tuple(_freeze(e) for e in v)
        if isinstance(v, dict):
            return tuple(sorted((k, _freeze(e)) for k, e in v.items() if k != "_w_plain"))
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            return ("array", tuple(v.shape), str(v.dtype), id(v))
        try:
            hash(v)
            return v
        except TypeError:
            return repr(v)

    return (func_name, _freeze(args), _freeze(kwargs or {}))


def _przs_np(shape, ptype="arithmetic"):
    import prix

    g0 = prix.generators.get("prev")
    g1 = prix.generators.get("next")
    u0 = g0.next_uint64(shape)
    u1 = g1.next_uint64(shape)
    diff = (u0 ^ u1) if ptype == "binary" else (u0 - u1)
    return diff.view(np.int64)


def _get_prng_offsets(n_przs, n_loc=0):
    import prix

    g0 = prix.generators.get("prev")
    g1 = prix.generators.get("next")
    s0, c0 = g0.next_offset(n_przs)
    s1, c1 = g1.next_offset(n_przs)
    if n_loc > 0:
        g_loc = prix.generators.get("local")
        s_loc, c_loc = g_loc.next_offset(n_loc)
    else:
        s_loc, c_loc = np.uint64(0), np.uint64(0)
    return c_loc, c0, c1, s_loc, s0, s1


@functools.partial(jax.jit, static_argnames=("s0_shape", "s1_shape", "c_shape", "is_rank0"))
def _tfp_mul_triple_jit(c_loc, c0, c1, s_loc, s0, s1, s0_shape, s1_shape, c_shape, is_rank0):
    n0 = int(np.prod(s0_shape)) if len(s0_shape) > 0 else 1
    n1 = int(np.prod(s1_shape)) if len(s1_shape) > 0 else 1
    n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
    tot = n0 + n1 + n2
    diff_all = _sm64_raw(s0, c0, (tot,)) - _sm64_raw(s1, c1, (tot,))
    diff_a = diff_all[:n0].reshape(s0_shape)
    diff_b = diff_all[n0 : n0 + n1].reshape(s1_shape)
    diff_c = diff_all[n0 + n1 :].reshape(c_shape)
    if is_rank0:
        ab = _sm64_raw(s_loc, c_loc, (n0 + n1,))
        a = ab[:n0].reshape(s0_shape)
        b = ab[n0:].reshape(s1_shape)
        c = a * b
        return diff_a + a, diff_b + b, diff_c + c
    return diff_a, diff_b, diff_c


@functools.partial(jax.jit, static_argnames=("s0_shape", "s1_shape", "c_shape", "is_rank0"))
def _tfp_matmul_triple_jit(c_loc, c0, c1, s_loc, s0, s1, s0_shape, s1_shape, c_shape, is_rank0):
    n0 = int(np.prod(s0_shape)) if len(s0_shape) > 0 else 1
    n1 = int(np.prod(s1_shape)) if len(s1_shape) > 0 else 1
    n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
    tot = n0 + n1 + n2
    diff_all = _sm64_raw(s0, c0, (tot,)) - _sm64_raw(s1, c1, (tot,))
    diff_a = diff_all[:n0].reshape(s0_shape)
    diff_b = diff_all[n0 : n0 + n1].reshape(s1_shape)
    diff_c = diff_all[n0 + n1 :].reshape(c_shape)
    if is_rank0:
        ab = _sm64_raw(s_loc, c_loc, (n0 + n1,))
        a = ab[:n0].reshape(s0_shape)
        b = ab[n0:].reshape(s1_shape)
        c = _int64_matmul(a, b)
        return diff_a + a, diff_b + b, diff_c + c
    return diff_a, diff_b, diff_c


@functools.partial(jax.jit, static_argnames=("s0_shape", "c_shape", "is_rank0"))
def _tfp_linear_matmul_jit(w_plain, c_loc, c0, c1, s_loc, s0, s1, s0_shape, c_shape, is_rank0):
    n0 = int(np.prod(s0_shape)) if len(s0_shape) > 0 else 1
    n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
    tot = n0 + n2
    diff_all = _sm64_raw(s0, c0, (tot,)) - _sm64_raw(s1, c1, (tot,))
    diff_a = diff_all[:n0].reshape(s0_shape)
    diff_c = diff_all[n0:].reshape(c_shape)
    if is_rank0:
        a = _sm64_raw(s_loc, c_loc, s0_shape)
        c = _int64_matmul(a, w_plain)
        return diff_a + a, diff_c + c
    return diff_a, diff_c


@functools.partial(jax.jit, static_argnames=("s0_shape", "c_shape", "is_rank0"))
def _tfp_linear_mul_jit(w_plain, c_loc, c0, c1, s_loc, s0, s1, s0_shape, c_shape, is_rank0):
    n0 = int(np.prod(s0_shape)) if len(s0_shape) > 0 else 1
    n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
    tot = n0 + n2
    diff_all = _sm64_raw(s0, c0, (tot,)) - _sm64_raw(s1, c1, (tot,))
    diff_a = diff_all[:n0].reshape(s0_shape)
    diff_c = diff_all[n0:].reshape(c_shape)
    if is_rank0:
        a = _sm64_raw(s_loc, c_loc, s0_shape)
        c = a * w_plain
        return diff_a + a, diff_c + c
    return diff_a, diff_c


@functools.partial(jax.jit, static_argnames=("shape", "is_rank0"))
def _tfp_mul_triple_and_pre_jit(x_share, y_share, c_loc, c0, c1, s_loc, s0, s1, shape, is_rank0):
    diff = _sm64_raw(s0, c0, (3, *shape)) - _sm64_raw(s1, c1, (3, *shape))
    if is_rank0:
        ab = _sm64_raw(s_loc, c_loc, (2, *shape))
        a_sh = diff[0] + ab[0]
        b_sh = diff[1] + ab[1]
        c_sh = diff[2] + ab[0] * ab[1]
    else:
        a_sh, b_sh, c_sh = diff[0], diff[1], diff[2]
    ed_share = jnp.stack([x_share - a_sh, y_share - b_sh], axis=0)
    return ed_share, a_sh, b_sh, c_sh


@functools.partial(jax.jit, static_argnames=("shape", "is_rank0"))
def _tfp_square_jit(c_loc, c0, c1, s_loc, s0, s1, shape, is_rank0):
    diff = _sm64_raw(s0, c0, (2, *shape)) - _sm64_raw(s1, c1, (2, *shape))
    if is_rank0:
        r = _sm64_raw(s_loc, c_loc, shape)
        return diff[0] + r, diff[1] + (r * r)
    return diff[0], diff[1]


@functools.partial(jax.jit, static_argnames=("shape", "is_rank0"))
def _tfp_square_and_pre_jit(x_share, c_loc, c0, c1, s_loc, s0, s1, shape, is_rank0):
    diff = _sm64_raw(s0, c0, (2, *shape)) - _sm64_raw(s1, c1, (2, *shape))
    if is_rank0:
        r = _sm64_raw(s_loc, c_loc, shape)
        r_sh = diff[0] + r
        r2_sh = diff[1] + (r * r)
    else:
        r_sh, r2_sh = diff[0], diff[1]
    return x_share - r_sh, r_sh, r2_sh


@functools.partial(jax.jit, static_argnames=("s0_shape", "s1_shape", "c_shape", "is_rank0"))
def _tfp_binary_triple_jit(c_loc, c0, c1, s_loc, s0, s1, s0_shape, s1_shape, c_shape, is_rank0):
    n0 = int(np.prod(s0_shape)) if len(s0_shape) > 0 else 1
    n1 = int(np.prod(s1_shape)) if len(s1_shape) > 0 else 1
    n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
    tot = n0 + n1 + n2
    diff_all = _sm64_raw(s0, c0, (tot,)) ^ _sm64_raw(s1, c1, (tot,))
    diff_a = diff_all[:n0].reshape(s0_shape)
    diff_b = diff_all[n0 : n0 + n1].reshape(s1_shape)
    diff_c = diff_all[n0 + n1 :].reshape(c_shape)
    if is_rank0:
        ab = _sm64_raw(s_loc, c_loc, (n0 + n1,))
        a = ab[:n0].reshape(s0_shape)
        b = ab[n0:].reshape(s1_shape)
        c = a & b
        return diff_a ^ a, diff_b ^ b, diff_c ^ c
    return diff_a, diff_b, diff_c


@functools.partial(jax.jit, static_argnames=("shape", "k_rounds", "is_rank0"))
def _tfp_spk_triples_jit(c_loc, c0, c1, s_loc, s0, s1, shape, k_rounds, is_rank0):
    a_shape = (k_rounds,) + shape
    b_shape = (k_rounds, 2) + shape
    n_a = int(np.prod(a_shape))
    n_b = int(np.prod(b_shape))
    tot = n_a + 2 * n_b
    diff_all = _sm64_raw(s0, c0, (tot,)) ^ _sm64_raw(s1, c1, (tot,))
    diff_a = diff_all[:n_a].reshape(a_shape)
    diff_b = diff_all[n_a : n_a + n_b].reshape(b_shape)
    diff_c = diff_all[n_a + n_b :].reshape(b_shape)
    if is_rank0:
        ab = _sm64_raw(s_loc, c_loc, (n_a + n_b,))
        a = ab[:n_a].reshape(a_shape)
        b = ab[n_a:].reshape(b_shape)
        c = a[:, None] & b
        return diff_a ^ a, diff_b ^ b, diff_c ^ c
    return diff_a, diff_b, diff_c


@functools.partial(jax.jit, static_argnames=("shape", "k_rounds", "is_rank0"))
def _tfp_ltz_rng_and_pre_jit(x_share, c_loc, c0, c1, s_loc, s0, s1, shape, k_rounds, is_rank0):
    n = int(np.prod(shape)) if len(shape) > 0 else 1
    a_shape = (k_rounds,) + shape
    b_shape = (k_rounds, 2) + shape
    n_a = k_rounds * n
    n_b = k_rounds * 2 * n
    tot_bin = 3 * n + n_a + 2 * n_b
    tot_all = tot_bin + 2 * n

    u0 = _sm64_raw(s0, c0, (tot_all,))
    u1 = _sm64_raw(s1, c1, (tot_all,))
    diff_xor = u0 ^ u1
    diff_rA = (u0[tot_bin : tot_bin + n] - u1[tot_bin : tot_bin + n]).reshape(shape)
    diff_rB = diff_xor[tot_bin + n : tot_bin + 2 * n].reshape(shape)

    diff_a0 = diff_xor[:n].reshape(shape)
    diff_b0 = diff_xor[n : 2 * n].reshape(shape)
    diff_c0 = diff_xor[2 * n : 3 * n].reshape(shape)
    off = 3 * n
    diff_a = diff_xor[off : off + n_a].reshape(a_shape)
    diff_b = diff_xor[off + n_a : off + n_a + n_b].reshape(b_shape)
    diff_c = diff_xor[off + n_a + n_b : tot_bin].reshape(b_shape)

    if is_rank0:
        loc = _sm64_raw(s_loc, c_loc, (2 * n + n_a + n_b + n,))
        a0 = loc[:n].reshape(shape)
        b0 = loc[n : 2 * n].reshape(shape)
        c0 = a0 & b0
        a_spk = loc[2 * n : 2 * n + n_a].reshape(a_shape)
        b_spk = loc[2 * n + n_a : 2 * n + n_a + n_b].reshape(b_shape)
        c_spk = a_spk[:, None] & b_spk
        r_bit = loc[2 * n + n_a + n_b :].reshape(shape) & jnp.int64(1)

        a0_sh = diff_a0 ^ a0
        b0_sh = diff_b0 ^ b0
        c0_sh = diff_c0 ^ c0
        a_all = diff_a ^ a_spk
        b_all = diff_b ^ b_spk
        c_all = diff_c ^ c_spk
        rA_sh = diff_rA + r_bit
        rB_sh = diff_rB ^ r_bit
        ed_share = jnp.stack([x_share ^ a0_sh, b0_sh], axis=0)
    else:
        a0_sh, b0_sh, c0_sh = diff_a0, diff_b0, diff_c0
        a_all, b_all, c_all = diff_a, diff_b, diff_c
        rA_sh, rB_sh = diff_rA, diff_rB
        ed_share = jnp.stack([a0_sh, x_share ^ b0_sh], axis=0)

    return ed_share, a0_sh, b0_sh, c0_sh, a_all, b_all, c_all, rA_sh, rB_sh


@functools.partial(jax.jit, static_argnames=("shape", "is_rank0"))
def _tfp_b2a_rng_jit(c_loc, c0, c1, s_loc, s0, s1, shape, is_rank0):
    u0 = _sm64_raw(s0, c0, (2, *shape))
    u1 = _sm64_raw(s1, c1, (2, *shape))
    diff_a = u0[0] - u1[0]
    diff_b = u0[1] ^ u1[1]
    if is_rank0:
        r = _sm64_raw(s_loc, c_loc, shape) & jnp.int64(1)
        return diff_a + r, diff_b ^ r
    return diff_a, diff_b


@functools.partial(jax.jit, static_argnames=("shape", "is_rank0"))
def _tfp_wrap_rng_jit(c_loc, c0, c1, s_loc, s0, s1, shape, is_rank0):
    diff = _sm64_raw(s0, c0, (2, *shape)) - _sm64_raw(s1, c1, (2, *shape))
    if is_rank0:
        r_loc = _sm64_raw(s_loc, c_loc, shape)
        r0 = diff[0] + r_loc
        r1 = -diff[0]
        theta_r = (
            ((r0 > 0) & (r1 > 0) & (r_loc < 0)).astype(DATATYPE)
            - ((r0 < 0) & (r1 < 0) & (r_loc > 0)).astype(DATATYPE)
        )
        return r0, diff[1] + theta_r
    return diff[0], diff[1]


@functools.partial(jax.jit, static_argnames=("t_shape", "lut_size", "is_rank0"))
def _tfp_one_hot_jit(c_loc, c0, c1, s_loc, s0, s1, t_shape, lut_size, is_rank0):
    n_elem = int(np.prod(t_shape)) if len(t_shape) > 0 else 1
    oh_shape = (n_elem, lut_size)
    tot = n_elem + n_elem * lut_size
    diff_all = _sm64_raw(s0, c0, (tot,)) - _sm64_raw(s1, c1, (tot,))
    diff_r = diff_all[:n_elem].reshape(t_shape)
    diff_oh = diff_all[n_elem:].reshape(oh_shape)
    if is_rank0:
        # The additive mask r must be (statistically) uniform over a range much
        # larger than |x| so that opening z = x - r hides x. We draw r uniformly
        # from [0, 2^61): this hides any |x| << 2^61 with statistical distance
        # |x| / 2^61, and since x - r never wraps mod 2^64, (x - r) mod lut_size
        # stays exact for every LUT size (power of two or not).
        r = _sm64_raw(s_loc, c_loc, t_shape) & jnp.int64((1 << 61) - 1)
        r_clear = r % jnp.int64(lut_size)
        r_share = diff_r + r
        oh_share = diff_oh.at[jnp.arange(n_elem), r_clear.reshape(-1)].add(1)
        return r_share, oh_share
    return diff_r, diff_oh


@functools.partial(jax.jit, static_argnames=("shape", "l_minus_m", "m", "is_rank0"))
def _tfp_egk_rng_jit(c_loc, c0, c1, s_loc, s0, s1, shape, l_minus_m, m, is_rank0):
    diff = _sm64_raw(s0, c0, (3, *shape)) - _sm64_raw(s1, c1, (3, *shape))
    if is_rank0:
        loc = _sm64_raw(s_loc, c_loc, (3, *shape))
        r = loc[0] & jnp.int64((1 << l_minus_m) - 1)
        rp = loc[1] & jnp.int64((1 << m) - 1)
        b = loc[2] & jnp.int64(1)
        return diff[0] + r, diff[1] + rp, diff[2] + b
    return diff[0], diff[1], diff[2]


@functools.partial(jax.jit, static_argnames=("shape", "l", "m", "is_rank0"))
def _tfp_egk_rng_and_pre_jit(x_share, c_loc, c0, c1, s_loc, s0, s1, shape, l, m, is_rank0):
    l_minus_m = l - m
    diff = _sm64_raw(s0, c0, (3, *shape)) - _sm64_raw(s1, c1, (3, *shape))
    if is_rank0:
        loc = _sm64_raw(s_loc, c_loc, (3, *shape))
        r_sh = diff[0] + (loc[0] & jnp.int64((1 << l_minus_m) - 1))
        rp_sh = diff[1] + (loc[1] & jnp.int64((1 << m) - 1))
        b_sh = diff[2] + (loc[2] & jnp.int64(1))
        a_p = x_share + (jnp.int64(1) << (l - 1))
    else:
        r_sh, rp_sh, b_sh = diff[0], diff[1], diff[2]
        a_p = x_share
    rpp = (jnp.int64(1) << m) * r_sh + rp_sh
    enc_c_share = (jnp.int64(1) << (64 - l - 1)) * (a_p + (jnp.int64(1) << l) * b_sh + rpp)
    return enc_c_share, r_sh, b_sh


@functools.partial(
    jax.jit,
    static_argnames=("shape", "period_scaled", "period_f", "terms", "scale", "is_rank0"),
)
def _tfp_trig_triple_jit(c_loc, c0, c1, s_loc, s0, s1, shape, period_scaled, period_f, terms, scale, is_rank0):
    n0 = int(np.prod(shape)) if len(shape) > 0 else 1
    uv_shape = (terms,) + shape
    n_uv = terms * n0
    tot = n0 + 2 * n_uv
    diff_all = _sm64_raw(s0, c0, (tot,)) - _sm64_raw(s1, c1, (tot,))
    diff_t = diff_all[:n0].reshape(shape)
    diff_u = diff_all[n0 : n0 + n_uv].reshape(uv_shape)
    diff_v = diff_all[n0 + n_uv :].reshape(uv_shape)
    if is_rank0:
        t = _sm64_raw(s_loc, c_loc, shape) & jnp.int64((1 << 61) - 1)
        t_mod = jnp.mod(t, jnp.int64(period_scaled)).astype(jnp.float32) / jnp.float32(scale)
        ks = jnp.arange(1, terms + 1, dtype=jnp.float32) * jnp.float32(2.0 * np.pi / period_f)
        angles = jnp.tensordot(ks, t_mod, axes=0)
        u = jnp.round(jnp.sin(angles) * jnp.float32(scale)).astype(DATATYPE)
        v = jnp.round(jnp.cos(angles) * jnp.float32(scale)).astype(DATATYPE)
        return diff_t + t, diff_u + u, diff_v + v
    return diff_t, diff_u, diff_v


def _wrap_shares(shares, ptypes, device=None, precision=0):
    from prix.array import Array, Ptype, SENTINEL

    out = []
    for s, pt in zip(shares, ptypes):
        res = Array(src=SENTINEL, precision=precision, device=device, ptype=Ptype(pt))
        res.share = to_jax(s, device=device, dtype=DATATYPE)
        out.append(res)
    return tuple(out)


def _share_batch_np(np_tensors, shapes, ptypes, device=None):
    from prix.array import Array, Ptype, SENTINEL

    rank = comm.get().get_rank()
    out = []
    for idx, (shape, ptype) in enumerate(zip(shapes, ptypes)):
        diff = _przs_np(shape, ptype)
        if rank == 0 and np_tensors[idx] is not None:
            if ptype == "binary":
                diff = diff ^ np_tensors[idx]
            else:
                diff = diff + np_tensors[idx]
        res = Array(src=SENTINEL, precision=0, device=device, ptype=Ptype(ptype))
        res.share = to_jax(diff, device=device, dtype=DATATYPE)
        out.append(res)
    return out


def _share_batch(tensors, shapes, ptypes, device=None):
    """Share a batch of dealer tensors from rank 0 to all parties via PRZS (zero IPC)."""
    from prix.array import Array

    rank = comm.get().get_rank()
    out = []
    for idx, (shape, ptype) in enumerate(zip(shapes, ptypes)):
        res = Array.PRZS(shape, device=device, ptype=ptype)
        res.precision = 0
        if rank == 0:
            if ptype == "binary":
                res.share = res.share ^ tensors[idx]
            else:
                res.share = res.share + tensors[idx]
        out.append(res)
    return out


class TrustedFirstParty(TupleProvider):
    """Trusted First Party (party 0) provider for correlated randomness."""

    NAME = "TFP"

    def generate_additive_triple(self, size0, size1, op="mul", device=None, *args, **kwargs):
        s0, s1 = tuple(size0), tuple(size1)
        is_rank0 = self.rank == 0
        if op == "mul":
            c_shape = jnp.broadcast_shapes(s0, s1)
            _acct_preproc(64 * (int(np.prod(s0)) + int(np.prod(s1)) + int(np.prod(c_shape))))
            if jax.default_backend() == "gpu":
                n0 = int(np.prod(s0)) if len(s0) > 0 else 1
                n1 = int(np.prod(s1)) if len(s1) > 0 else 1
                n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
                c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n0 + n1 + n2, n0 + n1 if is_rank0 else 0)
                shares = _tfp_mul_triple_jit(c_loc, c0, c1, s_loc, sd0, sd1, s0, s1, c_shape, is_rank0)
                return _wrap_shares(shares, ("arithmetic", "arithmetic", "arithmetic"), device=device)
            if is_rank0:
                a_np = generate_random_ring_element_np(s0, device=device)
                b_np = generate_random_ring_element_np(s1, device=device)
                c_np = a_np * b_np
                tensors = [a_np, b_np, c_np]
            else:
                tensors = [None, None, None]
            return tuple(_share_batch_np(tensors, (s0, s1, c_shape), ("arithmetic", "arithmetic", "arithmetic"), device=device))

        if op == "matmul":
            c_shape = jnp.broadcast_shapes(s0[:-2], s1[:-2]) + (s0[-2], s1[-1])
            n0 = int(np.prod(s0)) if len(s0) > 0 else 1
            n1 = int(np.prod(s1)) if len(s1) > 0 else 1
            n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
            _acct_preproc(64 * (n0 + n1 + n2))
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n0 + n1 + n2, n0 + n1 if is_rank0 else 0)
            shares = _tfp_matmul_triple_jit(c_loc, c0, c1, s_loc, sd0, sd1, s0, s1, c_shape, is_rank0)
            return _wrap_shares(shares, ("arithmetic", "arithmetic", "arithmetic"), device=device)

        if is_rank0:
            a = generate_random_ring_element(s0, device=device)
            b = generate_random_ring_element(s1, device=device)
            c = _eval_ring_op(op, a, b, *args, **kwargs)
            tensors = [a, b, c]
            c_shape = c.shape
        else:
            tensors = [None, None, None]
            a_dummy = jnp.zeros(s0, dtype=DATATYPE)
            b_dummy = jnp.zeros(s1, dtype=DATATYPE)
            c_shape = _eval_ring_op(op, a_dummy, b_dummy, *args, **kwargs).shape
        shapes = (s0, s1, c_shape)
        _acct_preproc(64 * sum(int(np.prod(s)) for s in shapes))
        return tuple(_share_batch(tensors, shapes, ("arithmetic", "arithmetic", "arithmetic"), device=device))

    def square(self, size, device=None):
        shape = tuple(size)
        is_rank0 = self.rank == 0
        _acct_preproc(64 * 2 * int(np.prod(shape)))
        if jax.default_backend() == "gpu":
            n = int(np.prod(shape)) if len(shape) > 0 else 1
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(2 * n, n if is_rank0 else 0)
            shares = _tfp_square_jit(c_loc, c0, c1, s_loc, sd0, sd1, shape, is_rank0)
            return _wrap_shares(shares, ("arithmetic", "arithmetic"), device=device)
        if is_rank0:
            r_np = generate_random_ring_element_np(shape, device=device)
            r2_np = r_np * r_np
            tensors = [r_np, r2_np]
        else:
            tensors = [None, None]
        return tuple(_share_batch_np(tensors, (shape, shape), ("arithmetic", "arithmetic"), device=device))

    def generate_binary_triple(self, size0, size1, device=None):
        s0, s1 = tuple(size0), tuple(size1)
        c_shape = jnp.broadcast_shapes(s0, s1)
        is_rank0 = self.rank == 0
        _acct_preproc(64 * (int(np.prod(s0)) + int(np.prod(s1)) + int(np.prod(c_shape))))
        if jax.default_backend() == "gpu":
            n0 = int(np.prod(s0)) if len(s0) > 0 else 1
            n1 = int(np.prod(s1)) if len(s1) > 0 else 1
            n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n0 + n1 + n2, n0 + n1 if is_rank0 else 0)
            shares = _tfp_binary_triple_jit(c_loc, c0, c1, s_loc, sd0, sd1, s0, s1, c_shape, is_rank0)
            return _wrap_shares(shares, ("binary", "binary", "binary"), device=device)
        if is_rank0:
            a_np = generate_kbit_random_tensor_np(s0, device=device)
            b_np = generate_kbit_random_tensor_np(s1, device=device)
            c_np = a_np & b_np
            tensors = [a_np, b_np, c_np]
        else:
            tensors = [None, None, None]
        return tuple(_share_batch_np(tensors, (s0, s1, c_shape), ("binary", "binary", "binary"), device=device))

    def generate_spk_triples(self, shape, k_rounds=6, device=None):
        s = tuple(shape)
        is_rank0 = self.rank == 0
        n = int(np.prod(s)) if len(s) > 0 else 1
        n_a = k_rounds * n
        n_b = k_rounds * 2 * n
        _acct_preproc(64 * (n_a + 2 * n_b))
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n_a + 2 * n_b, (n_a + n_b) if is_rank0 else 0)
        a_sh, b_sh, c_sh = _tfp_spk_triples_jit(c_loc, c0, c1, s_loc, sd0, sd1, s, k_rounds, is_rank0)
        return (
            to_jax(a_sh, device=device, dtype=DATATYPE),
            to_jax(b_sh, device=device, dtype=DATATYPE),
            to_jax(c_sh, device=device, dtype=DATATYPE),
        )

    def generate_ltz_rng_and_pre(self, x_share, k_rounds=6):
        s = tuple(x_share.shape)
        is_rank0 = self.rank == 0
        n = int(np.prod(s)) if len(s) > 0 else 1
        n_a = k_rounds * n
        n_b = k_rounds * 2 * n
        tot_przs = 5 * n + n_a + 2 * n_b
        tot_loc = (3 * n + n_a + n_b) if is_rank0 else 0
        _acct_preproc(64 * tot_przs)
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(tot_przs, tot_loc)
        return _tfp_ltz_rng_and_pre_jit(
            x_share, c_loc, c0, c1, s_loc, sd0, sd1, s, k_rounds, is_rank0
        )

    def generate_mul_triple_and_pre(self, x_share, y_share):
        shape = tuple(x_share.shape)
        is_rank0 = self.rank == 0
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        _acct_preproc(64 * 3 * n)
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(3 * n, 2 * n if is_rank0 else 0)
        return _tfp_mul_triple_and_pre_jit(
            x_share, y_share, c_loc, c0, c1, s_loc, sd0, sd1, shape, is_rank0
        )

    def square_and_pre(self, x_share):
        shape = tuple(x_share.shape)
        is_rank0 = self.rank == 0
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        _acct_preproc(64 * 2 * n)
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(2 * n, n if is_rank0 else 0)
        return _tfp_square_and_pre_jit(
            x_share, c_loc, c0, c1, s_loc, sd0, sd1, shape, is_rank0
        )

    def egk_rng_and_pre(self, x_share, l, m):
        shape = tuple(x_share.shape)
        is_rank0 = self.rank == 0
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        _acct_preproc((int(l) + 1) * n)  # r (l-m bits), r' (m bits), b (1 bit)
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(3 * n, 3 * n if is_rank0 else 0)
        return _tfp_egk_rng_and_pre_jit(
            x_share, c_loc, c0, c1, s_loc, sd0, sd1, shape, int(l), int(m), is_rank0
        )

    def wrap_rng(self, size, device=None):
        from prix.array import Array

        num_parties = comm.get().get_world_size()
        if num_parties == 2:
            shape = tuple(size)
            is_rank0 = self.rank == 0
            _acct_preproc(64 * 2 * int(np.prod(shape)))
            if jax.default_backend() == "gpu":
                n = int(np.prod(shape)) if len(shape) > 0 else 1
                c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(2 * n, n if is_rank0 else 0)
                shares = _tfp_wrap_rng_jit(c_loc, c0, c1, s_loc, sd0, sd1, shape, is_rank0)
                return _wrap_shares(shares, ("arithmetic", "arithmetic"), device=device)
            diff_r = _przs_np(shape, "arithmetic")
            diff_theta = _przs_np(shape, "arithmetic")
            if is_rank0:
                r_loc = generate_random_ring_element_np(shape, device=device)
                r0 = diff_r + r_loc
                r1 = -diff_r
                theta_r = (
                    ((r0 > 0) & (r1 > 0) & (r_loc < 0)).astype(np.int64)
                    - ((r0 < 0) & (r1 < 0) & (r_loc > 0)).astype(np.int64)
                )
                shares = (r0, diff_theta + theta_r)
            else:
                shares = (diff_r, diff_theta)
            return _wrap_shares(shares, ("arithmetic", "arithmetic"), device=device)

        if self.rank == 0:
            r = [
                generate_random_ring_element(size, device=device)
                for _ in range(num_parties)
            ]
            theta_r = count_wraps(r)
        else:
            r = None
            theta_r = jnp.zeros(size, dtype=DATATYPE)

        shares = comm.get().scatter(r, 0)
        r_shared = Array.from_shares(shares, precision=0, device=device)
        theta_r_shared = Array(theta_r, precision=0, src=0, device=device)
        return r_shared, theta_r_shared

    def B2A_rng(self, size, device=None):
        shape = tuple(size)
        is_rank0 = self.rank == 0
        _acct_preproc(65 * int(np.prod(shape)))  # 1 ring share + 1 binary share
        if jax.default_backend() == "gpu":
            n = int(np.prod(shape)) if len(shape) > 0 else 1
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(2 * n, n if is_rank0 else 0)
            shares = _tfp_b2a_rng_jit(c_loc, c0, c1, s_loc, sd0, sd1, shape, is_rank0)
            return _wrap_shares(shares, ("arithmetic", "binary"), device=device)
        if is_rank0:
            r_np = generate_kbit_random_tensor_np(shape, bitlength=1, device=device)
            tensors = [r_np, r_np]
        else:
            tensors = [None, None]
        return tuple(_share_batch_np(tensors, (shape, shape), ("arithmetic", "binary"), device=device))

    def generate_one_hot(self, tensor_size, lut_size, device=None):
        t_shape = tuple(tensor_size)
        n_elem = int(np.prod(t_shape)) if len(t_shape) > 0 else 1
        lut_sz = int(lut_size)
        is_rank0 = self.rank == 0
        _acct_preproc(64 * (n_elem + n_elem * lut_sz))
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n_elem + n_elem * lut_sz, n_elem if is_rank0 else 0)
        r_sh, oh_sh = _tfp_one_hot_jit(c_loc, c0, c1, s_loc, sd0, sd1, t_shape, lut_sz, is_rank0)
        return _wrap_shares((r_sh, oh_sh), ("arithmetic", "arithmetic"), device=device)

    def egk_trunc_pr_rng(self, size, l, m, device=None):
        shape = tuple(size)
        is_rank0 = self.rank == 0
        _acct_preproc((int(l) + 1) * int(np.prod(shape)))  # r (l-m bits), r' (m bits), b (1 bit)
        if jax.default_backend() == "gpu":
            n = int(np.prod(shape)) if len(shape) > 0 else 1
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(3 * n, 3 * n if is_rank0 else 0)
            shares = _tfp_egk_rng_jit(c_loc, c0, c1, s_loc, sd0, sd1, shape, int(l - m), int(m), is_rank0)
            return _wrap_shares(shares, ("arithmetic", "arithmetic", "arithmetic"), device=device)
        if is_rank0:
            r_np = generate_kbit_random_tensor_np(shape, bitlength=l - m, device=device)
            rp_np = generate_kbit_random_tensor_np(shape, bitlength=m, device=device)
            b_np = generate_kbit_random_tensor_np(shape, bitlength=1, device=device)
            tensors = [r_np, rp_np, b_np]
        else:
            tensors = [None, None, None]
        return tuple(_share_batch_np(tensors, (shape, shape, shape), ("arithmetic", "arithmetic", "arithmetic"), device=device))

    def generate_permutation(self, tensor_size, device=None):
        # Shared by all MPC parties (via the jointly-seeded "global" generator)
        # but unknown to the TTP / Fission evaluators; fresh on every call.
        import prix

        gen = prix.generators.get("global")
        if gen is None:
            raise RuntimeError("prix.init() must be called before generating permutations")
        return generate_permutation(tensor_size, generator=gen, device=device)

    def generate_trig_triple(self, size, period, terms, device=None):
        """Generate trigonometric triple (t, sin(k*t), cos(k*t)) for SHAFT Fourier series."""
        shape = tuple(size)
        n0 = int(np.prod(shape)) if len(shape) > 0 else 1
        terms_int = int(terms)
        n_uv = terms_int * n0
        prec = int(cfg.encoder.precision_bits)
        scale = 1 << prec
        period_f = float(period)
        period_scaled = int(round(period_f * scale))
        is_rank0 = self.rank == 0
        _acct_preproc(64 * (n0 + 2 * n_uv))
        c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(
            n0 + 2 * n_uv, n0 if is_rank0 else 0
        )
        shares = _tfp_trig_triple_jit(
            c_loc, c0, c1, s_loc, sd0, sd1, shape, period_scaled, period_f, terms_int, scale, is_rank0
        )
        return _wrap_shares(
            shares, ("arithmetic", "arithmetic", "arithmetic"), device=device, precision=prec
        )

    def generate_linear_triple(self, size0, size1, op="matmul", device=None, _w_plain=None, *args, **kwargs):
        """
        Generates a static-weight linear correlation (a, c = op(a, W)) where W is
        a model weight parameter of shape `size1`. Only `a` and `c` are shared;
        in the online phase, only `x - a` is opened (0 online bytes for W).
        """
        s0, s1 = tuple(size0), tuple(size1)
        is_rank0 = self.rank == 0
        if op == "matmul":
            c_shape = jnp.broadcast_shapes(s0[:-2], s1[:-2]) + (s0[-2], s1[-1])
            n0 = int(np.prod(s0)) if len(s0) > 0 else 1
            n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
            _acct_preproc(64 * (n0 + n2))
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n0 + n2, n0 if is_rank0 else 0)
            w_arg = _w_plain if (is_rank0 and _w_plain is not None) else jnp.zeros(s1, dtype=DATATYPE)
            shares = _tfp_linear_matmul_jit(
                w_arg, c_loc, c0, c1, s_loc, sd0, sd1, s0, c_shape, is_rank0
            )
            return _wrap_shares(shares, ("arithmetic", "arithmetic"), device=device)

        if op == "mul":
            c_shape = jnp.broadcast_shapes(s0, s1)
            n0 = int(np.prod(s0)) if len(s0) > 0 else 1
            n2 = int(np.prod(c_shape)) if len(c_shape) > 0 else 1
            _acct_preproc(64 * (n0 + n2))
            c_loc, c0, c1, s_loc, sd0, sd1 = _get_prng_offsets(n0 + n2, n0 if is_rank0 else 0)
            w_arg = _w_plain if (is_rank0 and _w_plain is not None) else jnp.zeros(s1, dtype=DATATYPE)
            shares = _tfp_linear_mul_jit(
                w_arg, c_loc, c0, c1, s_loc, sd0, sd1, s0, c_shape, is_rank0
            )
            return _wrap_shares(shares, ("arithmetic", "arithmetic"), device=device)

        if is_rank0:
            a = generate_random_ring_element(s0, device=device)
            w_arg = _w_plain if _w_plain is not None else jnp.zeros(s1, dtype=DATATYPE)
            c = _eval_ring_op(op, a, w_arg, *args, **kwargs)
            tensors = [a, c]
            c_shape = c.shape
        else:
            tensors = [None, None]
            a_dummy = jnp.zeros(s0, dtype=DATATYPE)
            b_dummy = jnp.zeros(s1, dtype=DATATYPE)
            c_shape = _eval_ring_op(op, a_dummy, b_dummy, *args, **kwargs).shape
        return tuple(_share_batch(tensors, (s0, c_shape), ("arithmetic", "arithmetic"), device=device))

    def generate_dcf_key(self, size, nbits=63, mode="ltz", device=None):
        """
        Generates correlated FSS/DCF comparison keys and B2A / DReLU-mul masks
        for `mode` in ('ltz', 'relu', 'ltz_abs_lt').
        """
        import prix
        from prix.fss import _tfp_dcf_key_jit

        shape = tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        is_rank0 = self.rank == 0
        g_glob = prix.generators.get("global")
        g0 = prix.generators.get("prev")
        g1 = prix.generators.get("next")
        s_glob, c_glob = g_glob.next_offset(4 * n)
        k_przs = 4 if mode == "ltz" else (5 if mode == "relu" else 7)
        s0, c0 = g0.next_offset(k_przs * n)
        s1, c1 = g1.next_offset(k_przs * n)
        # Key: nbits-level DPF CWs + r_in share + msb(r) bit + B2A masks (rA, rB) [+ w = rA * r_in].
        if mode == "ltz":
            _acct_fss_key(n, dpf_levels=int(nbits), ring_elems=2, bit_elems=2)
        elif mode == "relu":
            _acct_fss_key(n, dpf_levels=int(nbits), ring_elems=3, bit_elems=2)
        else:
            _acct_fss_key(n, dpf_levels=int(nbits), ring_elems=4, bit_elems=3)
        return _tfp_dcf_key_jit(
            c_glob, c0, c1, s_glob, s0, s1, shape, int(nbits), str(mode), is_rank0
        )

    def generate_sigma_tr_key(self, size, j_depth, device=None):
        """
        Generates the key for SIGMA's 1-round faithful Truncate-Reduce (Pi^TR, Theorem 4):
        shares of the mask `r` and of `r >> j_depth`, plus a single-path `j_depth`-level
        arithmetic DCF key for the carry `1{x_hat_lo < r_lo}`.
        """
        import prix
        from prix.fss import _tfp_sigma_tr_key_jit

        shape = tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        is_rank0 = self.rank == 0
        g_glob = prix.generators.get("global")
        g0 = prix.generators.get("prev")
        g1 = prix.generators.get("next")
        s_glob, c_glob = g_glob.next_offset(3 * n)
        s0, c0 = g0.next_offset(2 * n)
        s1, c1 = g1.next_offset(2 * n)
        _acct_fss_key(n, dcf_levels=int(j_depth), ring_elems=2)  # j-level arithmetic DCF + (r, r_hi) shares
        return _tfp_sigma_tr_key_jit(c_glob, c0, c1, s_glob, s0, s1, shape, int(j_depth), is_rank0)

    def generate_wave_key(self, size, J=5, j_depth=14, mode="bior", device=None):
        """
        Generates correlated Wave FSS keys (`mode` in ('haar', 'bior')):
          - 'haar': (J + j_depth)-level DPF key + Beaver multiplication triple (U, IP, U*IP).
          - 'bior': (J + j_depth)-level DPF key + j_depth-level arithmetic DCF key +
            Generalized Beaver 8-tuple (U, A, B, X, A*X - B, U*X, U*A, U*(A*X - B)).
        """
        import prix
        from prix.fss import _tfp_wave_key_jit

        shape = tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        is_rank0 = self.rank == 0
        g_glob = prix.generators.get("global")
        g0 = prix.generators.get("prev")
        g1 = prix.generators.get("next")
        mode_s = str(mode)
        k_glob = 5 if mode_s == "haar" else (7 if mode_s == "bior_prob" else 9)
        k_przs = 4 if mode_s == "haar" else (9 if mode_s == "bior_prob" else 10)
        s_glob, c_glob = g_glob.next_offset(k_glob * n)
        s0, c0 = g0.next_offset(k_przs * n)
        s1, c1 = g1.next_offset(k_przs * n)
        # Key: DPF CWs (+ DCF CWs for 'bior') + mask share(s) + Beaver tuple shares.
        if mode_s == "haar":
            _acct_fss_key(n, dpf_levels=int(J) + int(j_depth), ring_elems=4)
        elif mode_s == "bior_prob":
            _acct_fss_key(n, dpf_levels=int(J), ring_elems=9)
        else:
            _acct_fss_key(n, dpf_levels=int(J) + int(j_depth), dcf_levels=int(j_depth), ring_elems=10)
        return _tfp_wave_key_jit(
            c_glob, c0, c1, s_glob, s0, s1, shape, int(J), int(j_depth), mode_s, is_rank0
        )


class TrustedThirdParty(TupleProvider):
    """Trusted Third Party provider communicating with a dedicated TTPServer process."""

    NAME = "TTP"

    def __init__(self, device=None):
        super().__init__(device=device)

    def generate_additive_triple(self, size0, size1, op="mul", device=None, *args, **kwargs):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        a = generate_random_ring_element(size0, generator=generator, device=device)
        b = generate_random_ring_element(size1, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            c = TTPClient.get().ttp_request(
                "additive", device, size0, size1, op, *args, **kwargs
            )
        else:
            c_shape = _eval_ring_op(op, a, b, *args, **kwargs).shape
            c = generate_random_ring_element(c_shape, generator=generator, device=device)

        a = Array.from_shares(a, precision=0, device=device)
        b = Array.from_shares(b, precision=0, device=device)
        c = Array.from_shares(c, precision=0, device=device)
        return a, b, c

    def square(self, size, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            r2 = TTPClient.get().ttp_request("square", device, size)
        else:
            r2 = generate_random_ring_element(size, generator=generator, device=device)

        r = Array.from_shares(r, precision=0, device=device)
        r2 = Array.from_shares(r2, precision=0, device=device)
        return r, r2

    def generate_binary_triple(self, size0, size1, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        a = generate_kbit_random_tensor(size0, generator=generator, device=device)
        b = generate_kbit_random_tensor(size1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            c = TTPClient.get().ttp_request("binary", device, size0, size1)
        else:
            size2 = jnp.broadcast_shapes(tuple(size0), tuple(size1))
            c = generate_kbit_random_tensor(size2, generator=generator, device=device)

        a = Array.from_shares(a, precision=0, device=device, ptype="binary")
        b = Array.from_shares(b, precision=0, device=device, ptype="binary")
        c = Array.from_shares(c, precision=0, device=device, ptype="binary")
        return a, b, c

    def wrap_rng(self, size, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            theta_r = TTPClient.get().ttp_request("wraps", device, size)
        else:
            theta_r = generate_random_ring_element(size, generator=generator, device=device)

        r = Array.from_shares(r, precision=0, device=device)
        theta_r = Array.from_shares(theta_r, precision=0, device=device)
        return r, theta_r

    def B2A_rng(self, size, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        rB = generate_kbit_random_tensor(size, bitlength=1, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            rA = TTPClient.get().ttp_request("B2A", device, size)
        else:
            rA = generate_random_ring_element(size, generator=generator, device=device)

        rA = Array.from_shares(rA, precision=0, device=device, ptype="arithmetic")
        rB = Array.from_shares(rB, precision=0, device=device, ptype="binary")
        return rA, rB

    def generate_one_hot(self, tensor_size, lut_size, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(tensor_size, generator=generator, device=device)
        n_elem = int(np.prod(tensor_size)) if len(tensor_size) > 0 else 1
        if comm.get().get_rank() == 0:
            one_hot_r = TTPClient.get().ttp_request(
                "generate_one_hot", device, tensor_size, lut_size
            )
        else:
            one_hot_r = generate_random_ring_element(
                (n_elem, int(lut_size)), generator=generator, device=device
            )

        r = Array.from_shares(r, precision=0, device=device)
        one_hot_r = Array.from_shares(one_hot_r, precision=0, device=device)
        return r, one_hot_r

    def generate_permutation(self, tensor_size, device=None):
        # Shared by all MPC parties (via the jointly-seeded "global" generator)
        # but unknown to the TTP / Fission evaluators; fresh on every call.
        import prix

        if device is None:
            device = self.device
        gen = prix.generators.get("global")
        if gen is None:
            raise RuntimeError("prix.init() must be called before generating permutations")
        return generate_permutation(tensor_size, generator=gen, device=device)

    def egk_trunc_pr_rng(self, size, l, m, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)

        _ = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            r = TTPClient.get().ttp_request("egk_trunc_pr_rng", device, size, l - m)
        else:
            r = generate_random_ring_element(size, generator=generator, device=device)

        _ = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            r_p = TTPClient.get().ttp_request("egk_trunc_pr_rng", device, size, m)
        else:
            r_p = generate_random_ring_element(size, generator=generator, device=device)

        _ = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            b = TTPClient.get().ttp_request("egk_trunc_pr_rng", device, size, 1)
        else:
            b = generate_random_ring_element(size, generator=generator, device=device)

        r = Array.from_shares(r, precision=0, device=device)
        r_p = Array.from_shares(r_p, precision=0, device=device)
        b = Array.from_shares(b, precision=0, device=device)
        return r, r_p, b

    def generate_trig_triple(self, size, period, terms, device=None):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)
        prec = int(cfg.encoder.precision_bits)
        shape = tuple(size)
        uv_shape = (int(terms),) + shape

        _ = generate_random_ring_element(shape, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            t = TTPClient.get().ttp_request("trig_triple_t", device, shape, float(period), int(terms), prec)
            u = TTPClient.get().ttp_request("trig_triple_u", device, shape, float(period), int(terms), prec)
            v = TTPClient.get().ttp_request("trig_triple_v", device, shape, float(period), int(terms), prec)
        else:
            t = generate_random_ring_element(shape, generator=generator, device=device)
            u = generate_random_ring_element(uv_shape, generator=generator, device=device)
            v = generate_random_ring_element(uv_shape, generator=generator, device=device)

        return (
            Array.from_shares(t, precision=prec, device=device),
            Array.from_shares(u, precision=prec, device=device),
            Array.from_shares(v, precision=prec, device=device),
        )

    def generate_linear_triple(self, size0, size1, op="matmul", device=None, _w_plain=None, *args, **kwargs):
        from prix.array import Array

        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)
        s0, s1 = tuple(size0), tuple(size1)
        a = generate_random_ring_element(s0, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            w_arg = _w_plain if _w_plain is not None else jnp.zeros(s1, dtype=DATATYPE)
            c = TTPClient.get().ttp_request(
                "linear_triple", device, s0, s1, op, np.asarray(w_arg, dtype=np.int64), *args, **kwargs
            )
        else:
            b_dummy = jnp.zeros(s1, dtype=DATATYPE)
            c_shape = _eval_ring_op(op, a, b_dummy, *args, **kwargs).shape
            c = generate_random_ring_element(c_shape, generator=generator, device=device)
        return (
            Array.from_shares(a, precision=0, device=device),
            Array.from_shares(c, precision=0, device=device),
        )

    def generate_dcf_key(self, size, nbits=63, mode="ltz", device=None):
        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)
        shape = tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        nbits_i = int(nbits)
        mode_s = str(mode)

        r_in_sh = generate_random_ring_element(shape, generator=generator, device=device)
        s_raw = generate_random_ring_element((n,), generator=generator, device=device)
        s_init = jax.lax.bitcast_convert_type(s_raw, jnp.uint64) & ~jnp.uint64(3)
        msb_raw = generate_kbit_random_tensor((n,), bitlength=1, generator=generator, device=device)
        msb_r_sh = jax.lax.bitcast_convert_type(msb_raw, jnp.uint64)
        b2a_shape = ((2,) + shape) if mode_s == "ltz_abs_lt" else shape
        rB_sh = generate_kbit_random_tensor(b2a_shape, bitlength=1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            cw_i64 = TTPClient.get().ttp_request("dcf_cw", device, shape, nbits_i, mode_s)
            cw_all = jax.lax.bitcast_convert_type(cw_i64, jnp.uint64)
            msb_0_i64 = TTPClient.get().ttp_request("dcf_msb", device)
            msb_r_sh = jax.lax.bitcast_convert_type(msb_0_i64, jnp.uint64)
            rA_sh = TTPClient.get().ttp_request("dcf_rA", device)
            if mode_s in ("relu", "ltz_abs_lt"):
                w_sh = TTPClient.get().ttp_request("dcf_w", device)
        else:
            ttp_rank = comm.get().get_ttp_rank()
            old_v = comm.get().is_verbose()
            comm.get().set_verbosity(False)
            try:
                cw_i64 = comm.get().recv(src=ttp_rank)
            finally:
                comm.get().set_verbosity(old_v)
            cw_all = jax.lax.bitcast_convert_type(to_jax(cw_i64, device=device, dtype=DATATYPE), jnp.uint64)
            rA_sh = generate_random_ring_element(b2a_shape, generator=generator, device=device)
            if mode_s in ("relu", "ltz_abs_lt"):
                w_sh = generate_random_ring_element(shape, generator=generator, device=device)

        if mode_s == "ltz":
            return r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh
        return r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh, w_sh

    def generate_wave_key(self, size, J=5, j_depth=14, mode="bior", device=None):
        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)
        shape = tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        J_i = int(J)
        j_i = int(j_depth)
        mode_s = str(mode)

        r_sh = generate_random_ring_element((n,), generator=generator, device=device)
        s_dpf_raw = generate_random_ring_element((n,), generator=generator, device=device)
        s_dpf = jax.lax.bitcast_convert_type(s_dpf_raw, jnp.uint64) & ~jnp.uint64(3)

        if mode_s in ("haar", "bior_prob"):
            bvr_rows = 3 if mode_s == "haar" else 8
            if comm.get().get_rank() == 0:
                cw_dpf_i64 = TTPClient.get().ttp_request("wave_cw", device, shape, J_i, j_i, mode_s)
                cw_dpf = jax.lax.bitcast_convert_type(cw_dpf_i64, jnp.uint64)
                bvr_sh = TTPClient.get().ttp_request("wave_bvr", device)
            else:
                ttp_rank = comm.get().get_ttp_rank()
                old_v = comm.get().is_verbose()
                comm.get().set_verbosity(False)
                try:
                    cw_dpf_i64 = comm.get().recv(src=ttp_rank)
                finally:
                    comm.get().set_verbosity(old_v)
                cw_dpf = jax.lax.bitcast_convert_type(
                    to_jax(cw_dpf_i64, device=device, dtype=DATATYPE), jnp.uint64
                )
                bvr_sh = generate_random_ring_element((bvr_rows, n), generator=generator, device=device)
            return r_sh, s_dpf, cw_dpf, bvr_sh
        else:
            s_dcf_raw = generate_random_ring_element((n,), generator=generator, device=device)
            s_dcf = jax.lax.bitcast_convert_type(s_dcf_raw, jnp.uint64) & ~jnp.uint64(3)
            if comm.get().get_rank() == 0:
                cw_dpf_i64 = TTPClient.get().ttp_request("wave_cw", device, shape, J_i, j_i, mode_s)
                cw_dpf = jax.lax.bitcast_convert_type(cw_dpf_i64, jnp.uint64)
                r_lo_sh = TTPClient.get().ttp_request("wave_r_lo", device)
                cw_dcf_i64 = TTPClient.get().ttp_request("wave_dcf_cw", device)
                cw_dcf = jax.lax.bitcast_convert_type(cw_dcf_i64, jnp.uint64)
                v_cw_dcf = TTPClient.get().ttp_request("wave_dcf_v", device)
                g_cw_dcf = TTPClient.get().ttp_request("wave_dcf_g", device)
                bvr_sh = TTPClient.get().ttp_request("wave_bvr", device)
            else:
                ttp_rank = comm.get().get_ttp_rank()
                old_v = comm.get().is_verbose()
                comm.get().set_verbosity(False)
                try:
                    cw_dpf_i64 = comm.get().recv(src=ttp_rank)
                    cw_dcf_i64 = comm.get().recv(src=ttp_rank)
                    v_cw_dcf_np = comm.get().recv(src=ttp_rank)
                    g_cw_dcf_np = comm.get().recv(src=ttp_rank)
                finally:
                    comm.get().set_verbosity(old_v)
                cw_dpf = jax.lax.bitcast_convert_type(
                    to_jax(cw_dpf_i64, device=device, dtype=DATATYPE), jnp.uint64
                )
                cw_dcf = jax.lax.bitcast_convert_type(
                    to_jax(cw_dcf_i64, device=device, dtype=DATATYPE), jnp.uint64
                )
                v_cw_dcf = to_jax(v_cw_dcf_np, device=device, dtype=DATATYPE)
                g_cw_dcf = to_jax(g_cw_dcf_np, device=device, dtype=DATATYPE)
                r_lo_sh = generate_random_ring_element((n,), generator=generator, device=device)
                bvr_sh = generate_random_ring_element((8, n), generator=generator, device=device)
            return r_sh, r_lo_sh, s_dpf, cw_dpf, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf, bvr_sh

    def generate_sigma_tr_key(self, size, j_depth, device=None):
        if device is None:
            device = self.device
        generator = TTPClient.get().get_generator(device=device)
        shape = tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        j_i = int(j_depth)

        r_sh = generate_random_ring_element((n,), generator=generator, device=device)
        s_dcf_raw = generate_random_ring_element((n,), generator=generator, device=device)
        s_dcf = jax.lax.bitcast_convert_type(s_dcf_raw, jnp.uint64) & ~jnp.uint64(3)
        if comm.get().get_rank() == 0:
            cw_dcf_i64 = TTPClient.get().ttp_request("sigma_tr_cw", device, shape, j_i)
            cw_dcf = jax.lax.bitcast_convert_type(cw_dcf_i64, jnp.uint64)
            v_cw_dcf = TTPClient.get().ttp_request("sigma_tr_dcf_v", device)
            g_cw_dcf = TTPClient.get().ttp_request("sigma_tr_dcf_g", device)
            r_hi_sh = TTPClient.get().ttp_request("sigma_tr_r_hi", device)
        else:
            ttp_rank = comm.get().get_ttp_rank()
            old_v = comm.get().is_verbose()
            comm.get().set_verbosity(False)
            try:
                cw_dcf_i64 = comm.get().recv(src=ttp_rank)
                v_cw_dcf_np = comm.get().recv(src=ttp_rank)
                g_cw_dcf_np = comm.get().recv(src=ttp_rank)
            finally:
                comm.get().set_verbosity(old_v)
            cw_dcf = jax.lax.bitcast_convert_type(
                to_jax(cw_dcf_i64, device=device, dtype=DATATYPE), jnp.uint64
            )
            v_cw_dcf = to_jax(v_cw_dcf_np, device=device, dtype=DATATYPE)
            g_cw_dcf = to_jax(g_cw_dcf_np, device=device, dtype=DATATYPE)
            r_hi_sh = generate_random_ring_element((n,), generator=generator, device=device)
        return r_sh, r_hi_sh, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf

    @staticmethod
    def _init(device=None):
        TTPClient._init(device)

    @staticmethod
    def uninit():
        TTPClient.uninit()


class TTPClient:
    """Client on each MPC party that communicates with TTPServer."""

    __instance = None

    class _TTPClient:
        def __init__(self, device=None):
            self.ttp_group = comm.get().ttp_group
            self.comm_group = comm.get().ttp_comm_group
            self.device = device
            self._setup_generators(device)

        def _setup_generators(self, device=None):
            ttp_rank = comm.get().get_ttp_rank()
            seed = comm.get().recv(src=ttp_rank)
            self.generator = PRNGGenerator(seed=int(seed.item()), device=device)

        def get_generator(self, device=None):
            return self.generator

        def get_permutation_generator(self, device=None):
            import prix

            gen = prix.generators.get("global")
            if gen is None:
                raise RuntimeError("prix.init() must be called before generating permutations")
            return gen

        def ttp_request(self, func_name, device, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            if func_name not in TTPServer._ALLOWED_RPC_METHODS:
                raise ValueError(f"Disallowed TTP RPC method: {func_name!r}")
            message = {
                "function": func_name,
                "device": str(device) if device is not None else None,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()
            old_v = comm.get().is_verbose()
            comm.get().set_verbosity(False)
            try:
                comm.get().send_obj(message, ttp_rank, self.ttp_group)
                result = comm.get().recv(src=ttp_rank)
            finally:
                comm.get().set_verbosity(old_v)
            dev = resolve_device(device)
            if dev is not None:
                result = jax.device_put(result, dev)
            return result

    @staticmethod
    def _init(device=None):
        if TTPClient.__instance is None:
            TTPClient.__instance = TTPClient._TTPClient(device)

    @staticmethod
    def uninit():
        if TTPClient.__instance is not None and comm.is_initialized():
            if comm.get().get_rank() == 0:
                try:
                    comm.get().send_obj("terminate", comm.get().get_ttp_rank())
                except Exception:
                    pass
        TTPClient.__instance = None

    @staticmethod
    def get():
        if TTPClient.__instance is None:
            raise RuntimeError("TTPClient is not initialized")
        return TTPClient.__instance


class TTPServer:
    """Trusted Third Party server process."""

    _ALLOWED_RPC_METHODS = frozenset({
        "additive",
        "linear_triple",
        "dcf_cw",
        "dcf_msb",
        "dcf_rA",
        "dcf_w",
        "wave_cw",
        "wave_r_lo",
        "wave_dcf_cw",
        "wave_dcf_v",
        "wave_dcf_g",
        "wave_bvr",
        "sigma_tr_cw",
        "sigma_tr_dcf_v",
        "sigma_tr_dcf_g",
        "sigma_tr_r_hi",
        "square",
        "binary",
        "wraps",
        "B2A",
        "generate_one_hot",
        "egk_trunc_pr_rng",
        "trig_triple_t",
        "trig_triple_u",
        "trig_triple_v",
    })

    def __init__(self):
        self.ttp_group = comm.get().ttp_group
        self.comm_group = comm.get().ttp_comm_group
        self.device = "cpu"
        self._pending_trig_u = None
        self._pending_trig_v = None
        self._setup_generators()
        logging.info("TTPServer Initialized")

        try:
            while True:
                message = comm.get().recv_obj(0, self.ttp_group)
                if message == "terminate":
                    logging.info("TTPServer shutting down.")
                    return

                function = message["function"]
                if function not in self._ALLOWED_RPC_METHODS:
                    raise ValueError(f"Disallowed TTP RPC method: {function!r}")
                args = message["args"]
                kwargs = message["kwargs"]

                result = getattr(self, function)(*args, **kwargs)
                comm.get().send(result, 0)
        except Exception as err:
            logging.error(f"Encountered error. TTPServer shutting down: {err}")
            raise

    def _setup_generators(self):
        ws = comm.get().get_world_size()
        self.generators = []
        for i in range(ws):
            seed_i = int.from_bytes(os.urandom(8), "big") - 2**63
            comm.get().send(jnp.array(seed_i, dtype=DATATYPE), i)
            self.generators.append(PRNGGenerator(seed=seed_i))

    def _get_additive_PRSS(self, size, remove_rank=False):
        gens = self.generators[1:] if remove_rank else self.generators
        result = None
        for idx, g in enumerate(gens):
            elem = generate_random_ring_element(size, generator=g)
            result = elem if idx == 0 else result + elem
        return result

    def _get_binary_PRSS(self, size, bitlength=None, remove_rank=False):
        gens = self.generators[1:] if remove_rank else self.generators
        result = None
        for idx, g in enumerate(gens):
            elem = generate_kbit_random_tensor(size, bitlength=bitlength, generator=g)
            result = elem if idx == 0 else result ^ elem
        return result

    def additive(self, size0, size1, op, *args, **kwargs):
        a = self._get_additive_PRSS(size0)
        b = self._get_additive_PRSS(size1)
        c = _eval_ring_op(op, a, b, *args, **kwargs)
        c0 = c - self._get_additive_PRSS(c.shape, remove_rank=True)
        return c0

    def square(self, size):
        r = self._get_additive_PRSS(size)
        r2 = r * r
        return r2 - self._get_additive_PRSS(size, remove_rank=True)

    def binary(self, size0, size1):
        a = self._get_binary_PRSS(size0)
        b = self._get_binary_PRSS(size1)
        c = a & b
        return c ^ self._get_binary_PRSS(c.shape, remove_rank=True)

    def wraps(self, size):
        r = [generate_random_ring_element(size, generator=g) for g in self.generators]
        theta_r = count_wraps(r)
        return theta_r - self._get_additive_PRSS(size, remove_rank=True)

    def B2A(self, size):
        rB = self._get_binary_PRSS(size, bitlength=1)
        return rB - self._get_additive_PRSS(size, remove_rank=True)

    def generate_one_hot(self, tensor_size, lut_size):
        r = self._get_additive_PRSS(tensor_size)
        r_clear = r % int(lut_size)
        classes = jnp.arange(int(lut_size), dtype=DATATYPE)
        one_hot = (r_clear.reshape(-1, 1) == classes[None, :]).astype(DATATYPE)
        return one_hot - self._get_additive_PRSS(one_hot.shape, remove_rank=True)

    def egk_trunc_pr_rng(self, size, exp):
        raw = self._get_additive_PRSS(size)
        if exp <= 0:
            r = jnp.zeros_like(raw)
        elif exp >= 64:
            r = raw
        else:
            r = (raw.view(jnp.uint64) & jnp.uint64((1 << int(exp)) - 1)).astype(DATATYPE)
        return r - self._get_additive_PRSS(size, remove_rank=True)

    def trig_triple_t(self, size, period, terms, prec):
        scale = 1 << int(prec)
        period_scaled = int(round(float(period) * scale))
        raw = self._get_additive_PRSS(size)
        t = (raw.view(jnp.uint64) & jnp.uint64((1 << 61) - 1)).astype(DATATYPE)
        t_mod = jnp.mod(t, jnp.int64(period_scaled)).astype(jnp.float32) / jnp.float32(scale)
        ks = jnp.arange(1, int(terms) + 1, dtype=jnp.float32) * jnp.float32(2.0 * np.pi / float(period))
        angles = jnp.tensordot(ks, t_mod, axes=0)
        self._pending_trig_u = jnp.round(jnp.sin(angles) * jnp.float32(scale)).astype(DATATYPE)
        self._pending_trig_v = jnp.round(jnp.cos(angles) * jnp.float32(scale)).astype(DATATYPE)
        return t - self._get_additive_PRSS(size, remove_rank=True)

    def trig_triple_u(self, size, period, terms, prec):
        u = self._pending_trig_u
        self._pending_trig_u = None
        uv_shape = (int(terms),) + tuple(size)
        return u - self._get_additive_PRSS(uv_shape, remove_rank=True)

    def trig_triple_v(self, size, period, terms, prec):
        v = self._pending_trig_v
        self._pending_trig_v = None
        uv_shape = (int(terms),) + tuple(size)
        return v - self._get_additive_PRSS(uv_shape, remove_rank=True)

    def linear_triple(self, size0, size1, op, w_plain, *args, **kwargs):
        a = self._get_additive_PRSS(size0)
        w = jnp.asarray(w_plain, dtype=DATATYPE)
        c = _eval_ring_op(op, a, w, *args, **kwargs)
        return c - self._get_additive_PRSS(c.shape, remove_rank=True)

    def dcf_cw(self, shape, nbits, mode):
        from prix.fss import dpf_gen_jit

        shape = tuple(shape)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        nbits_i = int(nbits)
        mode_s = str(mode)

        r_in = self._get_additive_PRSS(shape)
        s0_raw = generate_random_ring_element((n,), generator=self.generators[0])
        s1_raw = generate_random_ring_element((n,), generator=self.generators[1])
        k0_0 = jax.lax.bitcast_convert_type(s0_raw, jnp.uint64) & ~jnp.uint64(3)
        k1_0 = jax.lax.bitcast_convert_type(s1_raw, jnp.uint64) & ~jnp.uint64(3)

        neg_r_u = jax.lax.bitcast_convert_type(-r_in.reshape(-1), jnp.uint64)
        low_mask = (
            (jnp.uint64(1) << jnp.uint64(nbits_i)) - jnp.uint64(1)
            if nbits_i < 64
            else ~jnp.uint64(0)
        )
        alpha = neg_r_u & low_mask
        msb_r = ((neg_r_u >> jnp.uint64(nbits_i)) & jnp.uint64(1)).astype(DATATYPE)
        _ = generate_kbit_random_tensor((n,), bitlength=1, generator=self.generators[0])
        msb_1 = generate_kbit_random_tensor((n,), bitlength=1, generator=self.generators[1])
        self._pending_dcf_msb = msb_r ^ msb_1

        cw_all = dpf_gen_jit(alpha, k0_0, k1_0, nbits_i)
        cw_i64 = jax.lax.bitcast_convert_type(cw_all, DATATYPE)
        comm.get().send(cw_i64, 1)

        b2a_shape = ((2,) + shape) if mode_s == "ltz_abs_lt" else shape
        rB = self._get_binary_PRSS(b2a_shape, bitlength=1)
        self._pending_dcf_rA = rB - self._get_additive_PRSS(b2a_shape, remove_rank=True)
        if mode_s in ("relu", "ltz_abs_lt"):
            r0 = rB[0] if mode_s == "ltz_abs_lt" else rB
            w = r0 * r_in
            self._pending_dcf_w = w - self._get_additive_PRSS(shape, remove_rank=True)
        return cw_i64

    def dcf_msb(self):
        res = self._pending_dcf_msb
        self._pending_dcf_msb = None
        return res

    def dcf_rA(self):
        res = self._pending_dcf_rA
        self._pending_dcf_rA = None
        return res

    def dcf_w(self):
        res = self._pending_dcf_w
        self._pending_dcf_w = None
        return res

    def wave_cw(self, shape, J, j_depth, mode):
        from prix.fss import dcf_gen_arith_jit, dpf_gen_jit

        shape = tuple(shape)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        J_i = int(J)
        j_i = int(j_depth)
        mode_s = str(mode)

        r_in = self._get_additive_PRSS((n,))
        r_u = jax.lax.bitcast_convert_type(r_in, jnp.uint64)
        s0_dpf_raw = generate_random_ring_element((n,), generator=self.generators[0])
        s1_dpf_raw = generate_random_ring_element((n,), generator=self.generators[1])
        k0_dpf = jax.lax.bitcast_convert_type(s0_dpf_raw, jnp.uint64) & ~jnp.uint64(3)
        k1_dpf = jax.lax.bitcast_convert_type(s1_dpf_raw, jnp.uint64) & ~jnp.uint64(3)

        two_J_mask = jnp.uint64((1 << J_i) - 1)
        two_j_mask = jnp.uint64((1 << j_i) - 1) if j_i > 0 else jnp.uint64(0)
        r_J = (r_u >> jnp.uint64(j_i)) & two_J_mask
        if j_i > 0:
            r_lo = r_u & two_j_mask
            alpha_lo = two_j_mask - r_lo
            alpha_full = (r_J << jnp.uint64(j_i)) | alpha_lo
        else:
            r_lo = jnp.zeros_like(r_u)
            alpha_lo = jnp.zeros_like(r_u)
            alpha_full = r_J

        cw_dpf = dpf_gen_jit(alpha_full, k0_dpf, k1_dpf, J_i + j_i)
        cw_dpf_i64 = jax.lax.bitcast_convert_type(cw_dpf, DATATYPE)
        comm.get().send(cw_dpf_i64, 1)

        if mode_s == "haar":
            U = generate_random_ring_element((n,))
            IP = generate_random_ring_element((n,))
            C = U * IP
            bvr_plain = jnp.stack([U, IP, C], axis=0)
            self._pending_wave_bvr = bvr_plain - self._get_additive_PRSS((3, n), remove_rank=True)
        elif mode_s == "bior_prob":
            U = generate_random_ring_element((n,))
            A = generate_random_ring_element((n,))
            B = generate_random_ring_element((n,))
            X = generate_random_ring_element((n,))
            AXmB = A * X - B
            UX = U * X
            UA = U * A
            UAXmB = U * AXmB
            bvr_plain = jnp.stack([U, A, B, X, AXmB, UX, UA, UAXmB], axis=0)
            self._pending_wave_bvr = bvr_plain - self._get_additive_PRSS((8, n), remove_rank=True)
        else:
            s0_dcf_raw = generate_random_ring_element((n,), generator=self.generators[0])
            s1_dcf_raw = generate_random_ring_element((n,), generator=self.generators[1])
            k0_dcf = jax.lax.bitcast_convert_type(s0_dcf_raw, jnp.uint64) & ~jnp.uint64(3)
            k1_dcf = jax.lax.bitcast_convert_type(s1_dcf_raw, jnp.uint64) & ~jnp.uint64(3)
            cw_dcf, v_cw_dcf, g_cw_dcf = dcf_gen_arith_jit(alpha_lo, k0_dcf, k1_dcf, j_i)
            cw_dcf_i64 = jax.lax.bitcast_convert_type(cw_dcf, DATATYPE)
            comm.get().send(cw_dcf_i64, 1)
            comm.get().send(v_cw_dcf, 1)
            comm.get().send(g_cw_dcf, 1)
            self._pending_wave_dcf_cw = cw_dcf_i64
            self._pending_wave_dcf_v = v_cw_dcf
            self._pending_wave_dcf_g = g_cw_dcf
            r_lo_i64 = jax.lax.bitcast_convert_type(r_lo, DATATYPE)
            self._pending_wave_r_lo = r_lo_i64 - self._get_additive_PRSS((n,), remove_rank=True)

            U = generate_random_ring_element((n,))
            A = generate_random_ring_element((n,))
            B = generate_random_ring_element((n,))
            X = generate_random_ring_element((n,))
            AXmB = A * X - B
            UX = U * X
            UA = U * A
            UAXmB = U * AXmB
            bvr_plain = jnp.stack([U, A, B, X, AXmB, UX, UA, UAXmB], axis=0)
            self._pending_wave_bvr = bvr_plain - self._get_additive_PRSS((8, n), remove_rank=True)
        return cw_dpf_i64

    def sigma_tr_cw(self, shape, j_depth):
        """Dealer side of SIGMA's Truncate-Reduce key: mirrors `TrustedThirdParty.generate_sigma_tr_key`."""
        from prix.fss import dcf_gen_arith_jit

        shape = tuple(shape)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        j_i = int(j_depth)
        r_in = self._get_additive_PRSS((n,))  # parties hold r_sh from their TTP generators
        r_u = jax.lax.bitcast_convert_type(r_in, jnp.uint64)
        s0_raw = generate_random_ring_element((n,), generator=self.generators[0])
        s1_raw = generate_random_ring_element((n,), generator=self.generators[1])
        k0 = jax.lax.bitcast_convert_type(s0_raw, jnp.uint64) & ~jnp.uint64(3)
        k1 = jax.lax.bitcast_convert_type(s1_raw, jnp.uint64) & ~jnp.uint64(3)
        two_j_mask = jnp.uint64((1 << j_i) - 1)
        r_lo = r_u & two_j_mask
        r_hi_i64 = jax.lax.bitcast_convert_type(r_u >> jnp.uint64(j_i), DATATYPE)
        cw_dcf, v_cw_dcf, g_cw_dcf = dcf_gen_arith_jit(r_lo, k0, k1, j_i)
        cw_dcf_i64 = jax.lax.bitcast_convert_type(cw_dcf, DATATYPE)
        comm.get().send(cw_dcf_i64, 1)
        comm.get().send(v_cw_dcf, 1)
        comm.get().send(g_cw_dcf, 1)
        self._pending_sigma_tr_dcf_v = v_cw_dcf
        self._pending_sigma_tr_dcf_g = g_cw_dcf
        self._pending_sigma_tr_r_hi = r_hi_i64 - self._get_additive_PRSS((n,), remove_rank=True)
        return cw_dcf_i64

    def sigma_tr_dcf_v(self):
        res = self._pending_sigma_tr_dcf_v
        self._pending_sigma_tr_dcf_v = None
        return res

    def sigma_tr_dcf_g(self):
        res = self._pending_sigma_tr_dcf_g
        self._pending_sigma_tr_dcf_g = None
        return res

    def sigma_tr_r_hi(self):
        res = self._pending_sigma_tr_r_hi
        self._pending_sigma_tr_r_hi = None
        return res

    def wave_r_lo(self):
        res = self._pending_wave_r_lo
        self._pending_wave_r_lo = None
        return res

    def wave_dcf_cw(self):
        res = self._pending_wave_dcf_cw
        self._pending_wave_dcf_cw = None
        return res

    def wave_dcf_v(self):
        res = self._pending_wave_dcf_v
        self._pending_wave_dcf_v = None
        return res

    def wave_dcf_g(self):
        res = self._pending_wave_dcf_g
        self._pending_wave_dcf_g = None
        return res

    def wave_bvr(self):
        res = self._pending_wave_bvr
        self._pending_wave_bvr = None
        return res


_PROVIDER_INSTANCES = {}


def get_default_provider(device=None):
    provider_name = str(cfg.mpc.provider).upper()
    if provider_name not in _PROVIDER_INSTANCES:
        if provider_name == "TFP":
            _PROVIDER_INSTANCES[provider_name] = TrustedFirstParty(device=device)
        elif provider_name == "TTP":
            _PROVIDER_INSTANCES[provider_name] = TrustedThirdParty(device=device)
        else:
            raise ValueError(f"Unknown MPC provider: {provider_name}")
    return _PROVIDER_INSTANCES[provider_name]


def set_default_provider(provider):
    if isinstance(provider, str):
        cfg.mpc.provider = provider.upper()
    elif isinstance(provider, TupleProvider):
        name = provider.NAME
        cfg.mpc.provider = name
        _PROVIDER_INSTANCES[name] = provider
    elif isinstance(provider, type) and issubclass(provider, TupleProvider):
        cfg.mpc.provider = provider.NAME
    else:
        raise ValueError(f"Invalid provider: {provider}")


def ttp_required():
    return str(cfg.mpc.provider).upper() == "TTP"
