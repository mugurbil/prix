#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import jax
import jax.numpy as jnp
import numpy as np

from prix.encoder import DATATYPE, resolve_device


class PRNGGenerator:
    """Stateful PRNG wrapper mimicking torch.Generator for MPC seed synchronization."""

    def __init__(self, seed=0, device=None):
        self.device = resolve_device(device)
        self._rng = None
        self.manual_seed(seed)

    def manual_seed(self, seed):
        if hasattr(seed, "item"):
            seed = int(seed.item())
        seed_u64 = int(seed) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed_u64))
        return self

    def next_uint64(self, shape):
        return self._rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=shape, dtype=np.uint64, endpoint=True)

    def next_permutation(self, n):
        return self._rng.permutation(int(n)).astype(np.int64)


def _normalize_shape(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    if len(size) == 1 and isinstance(size[0], int):
        return (size[0],)
    return tuple(size)


def random(seed, shape, device=None):
    """Deterministic random ring element generator from integer/array seed."""
    if isinstance(seed, PRNGGenerator):
        gen = seed
    else:
        gen = PRNGGenerator(seed=seed, device=device)
    bits = gen.next_uint64(tuple(shape))
    res = jnp.asarray(bits.view(np.int64), dtype=DATATYPE)
    dev = resolve_device(device)
    if dev is not None:
        res = jax.device_put(res, dev)
    return res


def _get_default_generator(generator=None, device=None):
    if generator is not None:
        return generator
    import prix

    local_gen = prix.generators.get("local")
    if isinstance(local_gen, PRNGGenerator):
        return local_gen
    dev_key = "cuda" if (device is not None and "cuda" in str(device).lower()) else "cpu"
    if isinstance(local_gen, dict) and dev_key in local_gen:
        return local_gen[dev_key]
    return PRNGGenerator(seed=np.random.randint(0, 2**31 - 1), device=device)


def generate_random_ring_element(*size, ring_size=(2**64), generator=None, device=None, **kwargs):
    """Generate a random element from the signed 64-bit ring Z_{2^64}."""
    shape = _normalize_shape(size)
    gen = _get_default_generator(generator, device)
    bits = gen.next_uint64(shape)
    if ring_size == 2**64:
        res = jnp.asarray(bits.view(np.int64), dtype=DATATYPE)
    else:
        res_np = (bits % np.uint64(ring_size)).astype(np.int64) - (ring_size // 2)
        res = jnp.asarray(res_np, dtype=DATATYPE)
    dev = resolve_device(device)
    if dev is not None:
        res = jax.device_put(res, dev)
    return res


def generate_kbit_random_tensor(*size, bitlength=None, generator=None, device=None, **kwargs):
    """Generate a random k-bit integer tensor in [0, 2^bitlength)."""
    shape = _normalize_shape(size)
    if bitlength is None or bitlength >= 64:
        return generate_random_ring_element(shape, generator=generator, device=device, **kwargs)
    if bitlength <= 0:
        res = jnp.zeros(shape, dtype=DATATYPE)
        dev = resolve_device(device)
        return jax.device_put(res, dev) if dev is not None else res

    gen = _get_default_generator(generator, device)
    bits = gen.next_uint64(shape)
    mask = np.uint64((1 << int(bitlength)) - 1)
    res = jnp.asarray((bits & mask).view(np.int64), dtype=DATATYPE)
    dev = resolve_device(device)
    if dev is not None:
        res = jax.device_put(res, dev)
    return res


def generate_permutation(size, generator=None, device=None, **kwargs):
    """Generate a random permutation and its inverse."""
    if isinstance(size, (tuple, list)):
        size = size[0]
    gen = _get_default_generator(generator, device)
    perm_np = gen.next_permutation(int(size))
    inv_perm_np = np.argsort(perm_np).astype(np.int64)
    permutation = jnp.asarray(perm_np, dtype=DATATYPE)
    inv_permutation = jnp.asarray(inv_perm_np, dtype=DATATYPE)
    dev = resolve_device(device)
    if dev is not None:
        permutation = jax.device_put(permutation, dev)
        inv_permutation = jax.device_put(inv_permutation, dev)
    return permutation, inv_permutation


@jax.jit
def _count_wraps_step(prev, cur, result):
    next_val = cur + prev
    result = result - ((prev < 0) & (cur < 0) & (next_val > 0)).astype(DATATYPE)
    result = result + ((prev > 0) & (cur > 0) & (next_val < 0)).astype(DATATYPE)
    return next_val, result


def count_wraps(share_list):
    """Computes the number of signed 64-bit overflows or underflows in a list of shares."""
    result = jnp.zeros_like(share_list[0], dtype=DATATYPE)
    prev = share_list[0]
    for cur in share_list[1:]:
        prev, result = _count_wraps_step(prev, cur, result)
    return result


@functools.lru_cache(maxsize=10)
def chebyshev_series(func, width, terms):
    """Computes Chebyshev coefficients for approximating `func` on [-width, width]."""
    n_range = jnp.arange(start=0, stop=terms, dtype=jnp.float32)
    x = width * jnp.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = jnp.cos(jnp.outer(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2.0 / terms) * jnp.sum(y * cos_term, axis=1)
    return coeffs
