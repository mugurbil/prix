#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import prix.communicator as comm
from prix.config import cfg
from prix.encoder import DATATYPE, resolve_device
from prix.random import (
    PRNGGenerator,
    count_wraps,
    generate_kbit_random_tensor,
    generate_permutation,
    generate_random_ring_element,
)


@jax.jit
def _int64_matmul(a, b):
    """
    Bit-exact 64-bit ring matrix multiplication mod 2^64 via limb splitting in float64:
    - For K <= 512: 3 limbs of 22 bits (6 float64 matmuls instead of CuRL's 9).
    - For K > 512:  4 limbs of 16 bits (10 float64 matmuls instead of CuRL's 16).
    """
    ua = jax.lax.bitcast_convert_type(a, jnp.uint64)
    ub = jax.lax.bitcast_convert_type(b, jnp.uint64)
    k = a.shape[-1]
    if k <= 128:
        mask22 = jnp.uint64((1 << 22) - 1)
        a0 = (ua & mask22).astype(jnp.float64)
        a1 = ((ua >> 22) & mask22).astype(jnp.float64)
        a2 = (ua >> 44).astype(jnp.float64)
        b0 = (ub & mask22).astype(jnp.float64)
        b1 = ((ub >> 22) & mask22).astype(jnp.float64)
        b2 = (ub >> 44).astype(jnp.float64)

        c0 = (a0 @ b0).astype(jnp.uint64)
        c1 = (a0 @ b1 + a1 @ b0).astype(jnp.float64).astype(jnp.uint64)
        c2 = (a0 @ b2 + a1 @ b1 + a2 @ b0).astype(jnp.float64).astype(jnp.uint64)
        res = c0 + (c1 << jnp.uint64(22)) + (c2 << jnp.uint64(44))
        return jax.lax.bitcast_convert_type(res, DATATYPE)
    elif k <= 512:
        mask22 = jnp.uint64((1 << 22) - 1)
        a0 = (ua & mask22).astype(jnp.float64)
        a1 = ((ua >> 22) & mask22).astype(jnp.float64)
        a2 = (ua >> 44).astype(jnp.float64)
        b0 = (ub & mask22).astype(jnp.float64)
        b1 = ((ub >> 22) & mask22).astype(jnp.float64)
        b2 = (ub >> 44).astype(jnp.float64)

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

        c0 = (a0 @ b0).astype(jnp.uint64)
        c1 = (a0 @ b1 + a1 @ b0).astype(jnp.uint64)
        c2 = (a0 @ b2 + a1 @ b1 + a2 @ b0).astype(jnp.uint64)
        c3 = (a0 @ b3 + a1 @ b2 + a2 @ b1 + a3 @ b0).astype(jnp.uint64)
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
        "square",
        "generate_binary_triple",
        "wrap_rng",
        "B2A_rng",
        "generate_one_hot",
        "egk_trunc_pr_rng",
    ]

    _DEFAULT_CACHE_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "tuple_cache")
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
        os.makedirs(prefix, exist_ok=True)
        return os.path.join(prefix, f"request_cache-{self.rank}.pkl")

    def _get_tuple_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
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
        with open(path, "wb") as f:
            pickle.dump(self.request_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.request_cache = []

    def _load_requests(self, filepath=None):
        path = self._get_request_path(prefix=filepath)
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.request_cache = pickle.load(f)

    def _save_tuples(self, filepath=None):
        if len(self.tuple_cache) == 0:
            return
        dirpath = self._get_tuple_path(prefix=filepath)
        os.makedirs(dirpath, exist_ok=True)
        existing_files = [
            f for f in os.listdir(dirpath) if f.startswith("tuple_batch_")
        ]
        next_index = len(existing_files)
        batch_file = os.path.join(dirpath, f"tuple_batch_{next_index}.pkl")
        tensor_cache = {k: list(v) for k, v in self.tuple_cache.items()}
        with open(batch_file, "wb") as f:
            pickle.dump(tensor_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
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
                batch_data = pickle.load(f)
            for key, values in batch_data.items():
                if key not in self.tuple_cache:
                    self.tuple_cache[key] = values

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
            hashable_request = (func_name, args)
            if hashable_request not in self.tuple_cache:
                self.tuple_cache[hashable_request] = result
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
            request = (func_name, args)
            if request in tuple_cache:
                return tuple(r.copy() for r in tuple_cache[request])
            return object.__getattribute__(self, func_name)(*args, **kwargs)

        return func_from_cache


@functools.partial(jax.jit, static_argnums=(1,))
def _make_one_hot(r, lut_size: int):
    r_clear = r % lut_size
    classes = jnp.arange(lut_size, dtype=DATATYPE)
    one_hot = (r_clear.reshape(-1, 1) == classes[None, :]).astype(DATATYPE)
    return r_clear, one_hot


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
        if self.rank == 0:
            a = generate_random_ring_element(size0, device=device)
            b = generate_random_ring_element(size1, device=device)
            c = _eval_ring_op(op, a, b, *args, **kwargs)
            tensors = [a, b, c]
            c_shape = c.shape
        else:
            tensors = [None, None, None]
            if op == "mul":
                c_shape = jnp.broadcast_shapes(tuple(size0), tuple(size1))
            elif op == "matmul":
                s0, s1 = tuple(size0), tuple(size1)
                c_shape = jnp.broadcast_shapes(s0[:-2], s1[:-2]) + (s0[-2], s1[-1])
            else:
                a_dummy = jnp.zeros(size0, dtype=DATATYPE)
                b_dummy = jnp.zeros(size1, dtype=DATATYPE)
                c_shape = _eval_ring_op(op, a_dummy, b_dummy, *args, **kwargs).shape
        shapes = (tuple(size0), tuple(size1), c_shape)
        return tuple(_share_batch(tensors, shapes, ("arithmetic", "arithmetic", "arithmetic"), device=device))

    def square(self, size, device=None):
        shape = tuple(size)
        if self.rank == 0:
            r = generate_random_ring_element(shape, device=device)
            r2 = r * r
            tensors = [r, r2]
        else:
            tensors = [None, None]
        return tuple(_share_batch(tensors, (shape, shape), ("arithmetic", "arithmetic"), device=device))

    def generate_binary_triple(self, size0, size1, device=None):
        s0, s1 = tuple(size0), tuple(size1)
        c_shape = jnp.broadcast_shapes(s0, s1)
        if self.rank == 0:
            a = generate_kbit_random_tensor(s0, device=device)
            b = generate_kbit_random_tensor(s1, device=device)
            c = a & b
            tensors = [a, b, c]
        else:
            tensors = [None, None, None]
        return tuple(_share_batch(tensors, (s0, s1, c_shape), ("binary", "binary", "binary"), device=device))

    def wrap_rng(self, size, device=None):
        from prix.array import Array

        if self.rank == 0:
            num_parties = comm.get().get_world_size()
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
        if self.rank == 0:
            r = generate_kbit_random_tensor(shape, bitlength=1, device=device)
            tensors = [r, r]
        else:
            tensors = [None, None]
        return tuple(_share_batch(tensors, (shape, shape), ("arithmetic", "binary"), device=device))

    def generate_one_hot(self, tensor_size, lut_size, device=None):
        t_shape = tuple(tensor_size)
        n_elem = int(np.prod(t_shape)) if len(t_shape) > 0 else 1
        oh_shape = (n_elem, int(lut_size))
        if self.rank == 0:
            r = generate_random_ring_element(t_shape, device=device)
            r_clear, one_hot = _make_one_hot(r, int(lut_size))
            tensors = [r_clear, one_hot]
        else:
            tensors = [None, None]
        return tuple(_share_batch(tensors, (t_shape, oh_shape), ("arithmetic", "arithmetic"), device=device))

    def egk_trunc_pr_rng(self, size, l, m, device=None):
        shape = tuple(size)
        if self.rank == 0:
            r = generate_kbit_random_tensor(shape, bitlength=l - m, device=device)
            r_p = generate_kbit_random_tensor(shape, bitlength=m, device=device)
            b = generate_kbit_random_tensor(shape, bitlength=1, device=device)
            tensors = [r, r_p, b]
        else:
            tensors = [None, None, None]
        return tuple(_share_batch(tensors, (shape, shape, shape), ("arithmetic", "arithmetic", "arithmetic"), device=device))

    def generate_permutation(self, tensor_size, device=None):
        gen = PRNGGenerator(seed=0, device=device)
        return generate_permutation(tensor_size, generator=gen, device=device)


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
        if device is None:
            device = self.device
        generator = TTPClient.get().get_permutation_generator(device=device)
        return generate_permutation(tensor_size, generator=generator, device=device)

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

            perm_seed = comm.get().recv(src=ttp_rank)
            self.permutation_generator = PRNGGenerator(
                seed=int(perm_seed.item()), device=device
            )

        def get_generator(self, device=None):
            return self.generator

        def get_permutation_generator(self, device=None):
            return self.permutation_generator

        def ttp_request(self, func_name, device, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            message = {
                "function": func_name,
                "device": str(device) if device is not None else None,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()
            comm.get().send_obj(message, ttp_rank, self.ttp_group)
            result = comm.get().recv(src=ttp_rank)
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

    def __init__(self):
        self.ttp_group = comm.get().ttp_group
        self.comm_group = comm.get().ttp_comm_group
        self.device = "cpu"
        self._setup_generators()
        logging.info("TTPServer Initialized")

        try:
            while True:
                message = comm.get().recv_obj(0, self.ttp_group)
                if message == "terminate":
                    logging.info("TTPServer shutting down.")
                    return

                function = message["function"]
                args = message["args"]
                kwargs = message["kwargs"]

                result = getattr(self, function)(*args, **kwargs)
                comm.get().send(result, 0)
        except RuntimeError as err:
            logging.info(f"Encountered Runtime error. TTPServer shutting down: {err}")

    def _setup_generators(self):
        ws = comm.get().get_world_size()
        self.generators = []
        for i in range(ws):
            seed_i = int.from_bytes(os.urandom(8), "big") - 2**63
            comm.get().send(jnp.array(seed_i, dtype=DATATYPE), i)
            self.generators.append(PRNGGenerator(seed=seed_i))

        perm_seed = int.from_bytes(os.urandom(8), "big") - 2**63
        for i in range(ws):
            comm.get().send(jnp.array(perm_seed, dtype=DATATYPE), i)

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
