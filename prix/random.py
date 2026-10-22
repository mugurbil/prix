#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import secrets

import jax
import jax.numpy as jnp
import numpy as np

from prix.encoder import DATATYPE, resolve_device, to_jax


class PRNGGenerator:
    """Stateful PRNG wrapper mimicking torch.Generator for MPC seed synchronization."""

    def __init__(self, seed=0, device=None):
        self.device = resolve_device(device)
        self._rng = None
        self._seed_u64 = 0
        self._counter = 0
        self.manual_seed(seed)

    def manual_seed(self, seed):
        if hasattr(seed, "item"):
            seed = int(seed.item())
        self._seed_u64 = int(seed) & 0xFFFFFFFFFFFFFFFF
        self._counter = 0
        self._rng = np.random.Generator(np.random.PCG64(self._seed_u64))
        return self

    def next_offset(self, count=1):
        c = self._counter
        self._counter += int(count)
        return self._seed_u64, c

    def next_uint64(self, shape):
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        self._counter += n
        return self._rng.integers(0, 0xFFFFFFFFFFFFFFFF, size=shape, dtype=np.uint64, endpoint=True)

    def next_permutation(self, n):
        self._counter += int(n)
        return self._rng.permutation(int(n)).astype(np.int64)


def _sm64_raw(seed_u64, offset_u64, shape):
    """Device-native SplitMix64 counter-based PRNG for inside @jax.jit kernels."""
    n = int(np.prod(shape)) if len(shape) > 0 else 1
    z = (
        jnp.uint64(seed_u64)
        + (jnp.uint64(offset_u64) + jnp.arange(n, dtype=jnp.uint64))
        * jnp.uint64(0x9E3779B97F4A7C15)
    )
    z = (z ^ (z >> jnp.uint64(30))) * jnp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> jnp.uint64(27))) * jnp.uint64(0x94D049BB133111EB)
    z = z ^ (z >> jnp.uint64(31))
    return jax.lax.bitcast_convert_type(z.reshape(shape), DATATYPE)


@functools.partial(jax.jit, static_argnames=("seed_u64", "shape", "bitlength"))
def _generate_kbit_jit(offset_u64, seed_u64, shape, bitlength):
    val = _sm64_raw(seed_u64, offset_u64, shape)
    if bitlength is not None and bitlength < 64:
        mask = jnp.int64((1 << int(bitlength)) - 1)
        val = val & mask
    return val


@functools.partial(jax.jit, static_argnames=("s0", "s1", "shape", "is_binary"))
def _przs_jit(c0, c1, s0, s1, shape, is_binary):
    u0 = _sm64_raw(s0, c0, shape)
    u1 = _sm64_raw(s1, c1, shape)
    return (u0 ^ u1) if is_binary else (u0 - u1)


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
    return to_jax(bits.view(np.int64), device=device, dtype=DATATYPE)


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
    return PRNGGenerator(seed=secrets.randbits(64), device=device)


def generate_random_ring_element_np(*size, ring_size=(2**64), generator=None, device=None):
    shape = _normalize_shape(size)
    gen = _get_default_generator(generator, device)
    bits = gen.next_uint64(shape)
    if ring_size == 2**64:
        return bits.view(np.int64)
    return (bits % np.uint64(ring_size)).astype(np.int64) - (ring_size // 2)


def generate_kbit_random_tensor_np(*size, bitlength=None, generator=None, device=None):
    shape = _normalize_shape(size)
    if bitlength is None or bitlength >= 64:
        return generate_random_ring_element_np(shape, generator=generator, device=device)
    if bitlength <= 0:
        return np.zeros(shape, dtype=np.int64)
    gen = _get_default_generator(generator, device)
    bits = gen.next_uint64(shape)
    mask = np.uint64((1 << int(bitlength)) - 1)
    return (bits & mask).view(np.int64)


def generate_random_ring_element(*size, ring_size=(2**64), generator=None, device=None, **kwargs):
    """Generate a random element from the signed 64-bit ring Z_{2^64}."""
    shape = _normalize_shape(size)
    if ring_size == 2**64 and jax.default_backend() == "gpu":
        gen = _get_default_generator(generator, device)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        seed_u64, offset_u64 = gen.next_offset(n)
        return to_jax(_generate_kbit_jit(offset_u64, seed_u64, shape, None), device=device, dtype=DATATYPE)
    arr_np = generate_random_ring_element_np(*size, ring_size=ring_size, generator=generator, device=device)
    return to_jax(arr_np, device=device, dtype=DATATYPE)


def generate_kbit_random_tensor(*size, bitlength=None, generator=None, device=None, **kwargs):
    """Generate a random k-bit integer tensor in [0, 2^bitlength)."""
    shape = _normalize_shape(size)
    if (bitlength is None or bitlength > 0) and jax.default_backend() == "gpu":
        gen = _get_default_generator(generator, device)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        seed_u64, offset_u64 = gen.next_offset(n)
        bl = None if (bitlength is None or bitlength >= 64) else int(bitlength)
        return to_jax(_generate_kbit_jit(offset_u64, seed_u64, shape, bl), device=device, dtype=DATATYPE)
    arr_np = generate_kbit_random_tensor_np(*size, bitlength=bitlength, generator=generator, device=device)
    return to_jax(arr_np, device=device, dtype=DATATYPE)


def generate_permutation(size, generator=None, device=None, **kwargs):
    """Generate a random permutation and its inverse."""
    if isinstance(size, (tuple, list)):
        size = size[0]
    gen = _get_default_generator(generator, device)
    perm_np = gen.next_permutation(int(size))
    inv_perm_np = _invert_permutation(perm_np)
    return to_jax(perm_np, device=device, dtype=DATATYPE), to_jax(inv_perm_np, device=device, dtype=DATATYPE)


def _invert_permutation(perm_np):
    """O(n) inverse of a permutation (cheaper than argsort's O(n log n))."""
    inv = np.empty_like(perm_np)
    inv[perm_np] = np.arange(perm_np.shape[0], dtype=perm_np.dtype)
    return inv


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


@functools.lru_cache(maxsize=16)
def fourier_series(func, width, terms, step=1e-4):
    """Computes Fourier series coefficients (alpha, beta_cos, beta_sin) on [-width, width]."""
    n_range = np.arange(1, int(terms) + 1, dtype=np.float64)
    x = np.arange(-float(width), float(width), float(step), dtype=np.float64)
    npx = np.pi * np.outer(n_range, x) / float(width)
    y = np.asarray(func(x), dtype=np.float64)
    cos_npx = np.cos(npx)
    sin_npx = np.sin(npx)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    alpha = float(trapz(y, x) / (2.0 * float(width)))
    beta_cos = jnp.asarray(trapz(y[None, :] * cos_npx, x, axis=-1) / float(width), dtype=jnp.float32)
    beta_sin = jnp.asarray(trapz(y[None, :] * sin_npx, x, axis=-1) / float(width), dtype=jnp.float32)
    return alpha, beta_cos, beta_sin
