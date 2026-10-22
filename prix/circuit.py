#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math

import jax
import jax.numpy as jnp

from prix.encoder import DATATYPE, resolve_device

__BITS = 64
__LOG_BITS = int(math.log2(__BITS))


@functools.lru_cache(maxsize=None)
def _spk_circuit_constants():
    """
    Generate the __MASKS, __OUT_MASKS, and __MULTIPLIERS constants
    used by _spk_circuit.
    """
    masks = jnp.array(
        [
            6148914691236517205,
            2459565876494606882,
            578721382704613384,
            36029346783166592,
            140737488388096,
            2147483648,
        ],
        dtype=DATATYPE,
    )
    multipliers = jnp.array(
        [(1 << (2**i + 1)) - 2 for i in range(__LOG_BITS)],
        dtype=DATATYPE,
    )
    out_masks = masks * multipliers
    return masks, out_masks, multipliers


@jax.jit
def _spk_pre_and_mask(sp_share, a_share, b_share, in_mask, out_mask, multiplier):
    p0_share = sp_share[1] & out_mask
    s1p1_share = (sp_share & in_mask) * multiplier
    e_share = p0_share ^ a_share
    f_share = s1p1_share ^ b_share
    return jnp.concatenate([e_share[None], f_share], axis=0)


@functools.partial(jax.jit, static_argnames=("is_rank0",))
def _spk_step_fused(
    sp_share,
    ef_prev,
    c_prev,
    a_prev,
    b_prev,
    out_mask_prev,
    a_next,
    b_next,
    in_mask_next,
    out_mask_next,
    mult_next,
    is_rank0,
):
    eps = ef_prev[0]
    delta = ef_prev[1:]
    b_eff = (b_prev ^ delta) if is_rank0 else b_prev
    update_share = c_prev ^ (b_eff & eps) ^ (a_prev & delta)
    not_out_mask = out_mask_prev ^ jnp.int64(-1)
    s_new = sp_share[0] ^ update_share[0]
    p_new = (sp_share[1] & not_out_mask) ^ update_share[1]
    sp_new = jnp.stack([s_new, p_new], axis=0)

    p0_share = p_new & out_mask_next
    s1p1_share = (sp_new & in_mask_next) * mult_next
    e_next = p0_share ^ a_next
    f_next = s1p1_share ^ b_next
    ef_next = jnp.concatenate([e_next[None], f_next], axis=0)
    return sp_new, ef_next


@functools.partial(jax.jit, static_argnames=("is_rank0",))
def _spk_combine_and_post(sp_share, c_share, ef, a_share, b_share, out_mask, is_rank0):
    eps = ef[0]
    delta = ef[1:]
    b_eff = (b_share ^ delta) if is_rank0 else b_share
    update_share = c_share ^ (b_eff & eps) ^ (a_share & delta)
    not_out_mask = out_mask ^ jnp.int64(-1)
    s_new = sp_share[0] ^ update_share[0]
    p_new = (sp_share[1] & not_out_mask) ^ update_share[1]
    return jnp.stack([s_new, p_new], axis=0)


def _spk_circuit(S, P):
    """
    Computes the Set-Propagate-Kill Tree circuit for a set (S, P)
    (K is implied by S, P since (SPK) is one-hot).

    See section 6.3 of Damgard et al., "Unconditionally Secure Constant-Rounds
    Multi-Party Computation for Equality, Comparison, Bits and Exponentiation".
    """
    import prix.communicator as comm
    from prix.array import Array, Ptype, SENTINEL
    from prix.provider import get_default_provider

    masks, out_masks, multipliers = _spk_circuit_constants()
    provider = get_default_provider()
    s_shape = S.size()
    is_rank0 = comm.get().get_rank() == 0
    sp_share = jnp.stack([S.share, P.share], axis=0)

    if hasattr(provider, "generate_spk_triples") and not provider.tracing and len(provider.tuple_cache) == 0:
        a_all, b_all, c_all = provider.generate_spk_triples(
            s_shape, k_rounds=__LOG_BITS, device=S.device
        )
        ef_share = _spk_pre_and_mask(
            sp_share, a_all[0], b_all[0], masks[0], out_masks[0], multipliers[0]
        )
        for i in range(__LOG_BITS - 1):
            ef = comm.get().all_reduce(ef_share, op="bxor")
            sp_share, ef_share = _spk_step_fused(
                sp_share,
                ef,
                c_all[i],
                a_all[i],
                b_all[i],
                out_masks[i],
                a_all[i + 1],
                b_all[i + 1],
                masks[i + 1],
                out_masks[i + 1],
                multipliers[i + 1],
                is_rank0,
            )
        ef_last = comm.get().all_reduce(ef_share, op="bxor")
        sp_share = _spk_combine_and_post(
            sp_share,
            c_all[__LOG_BITS - 1],
            ef_last,
            a_all[__LOG_BITS - 1],
            b_all[__LOG_BITS - 1],
            out_masks[__LOG_BITS - 1],
            is_rank0,
        )
    else:
        sp_shape = (2,) + s_shape
        for i in range(__LOG_BITS):
            a, b, c = provider.generate_binary_triple(s_shape, sp_shape, device=S.device)
            ef_share = _spk_pre_and_mask(
                sp_share, a.share, b.share, masks[i], out_masks[i], multipliers[i]
            )
            ef = comm.get().all_reduce(ef_share, op="bxor")
            sp_share = _spk_combine_and_post(
                sp_share, c.share, ef, a.share, b.share, out_masks[i], is_rank0
            )

    s_out = Array(src=SENTINEL, precision=0, device=S.device, ptype=Ptype.binary)
    s_out.share = sp_share[0]
    p_out = Array(src=SENTINEL, precision=0, device=S.device, ptype=Ptype.binary)
    p_out.share = sp_share[1]
    return s_out, p_out


def _p_circuit(P):
    """
    Computes the Propagate Tree circuit for input P.
    Returns sign bit 1 only if the binary of the input is all ones (-1).
    """
    shift = __BITS // 2
    for _ in range(__LOG_BITS):
        P = P & (P << shift)
        shift //= 2
    return P


def _flip_sign_bit(x):
    return x ^ jnp.int64(-9223372036854775808)


def _get_sign_bit(x):
    from prix.array import Array

    y = x >> 63
    if isinstance(y, Array):
        y.share = (y.share == -1).astype(DATATYPE)
    else:
        y = (y == -1).astype(DATATYPE)
    return y


def add(x, y):
    """Returns x + y from binary-shared Arrays `x` and `y`."""
    S = x & y
    P = x ^ y
    carry, _ = _spk_circuit(S, P)
    return P ^ (carry << 1)


def eq(x, y):
    """Returns x == y from binary-shared Arrays `x` and `y`."""
    bitwise_equal = ~(x ^ y)
    P = _p_circuit(bitwise_equal)
    return _get_sign_bit(P)


def ne(x, y):
    """Returns x != y from binary-shared Arrays `x` and `y`."""
    return eq(x, y) ^ 1


def lt(x, y):
    """Returns x < y from binary-shared Arrays `x` and `y`."""
    x, y = _flip_sign_bit(x), _flip_sign_bit(y)
    S = y & ~x
    P = ~(x ^ y)
    S, _ = _spk_circuit(S, P)
    return _get_sign_bit(S)


def le(x, y):
    """Returns x <= y from binary-shared Arrays `x` and `y`."""
    x, y = _flip_sign_bit(x), _flip_sign_bit(y)
    S = y & ~x
    P = ~(x ^ y)
    S, P = _spk_circuit(S, P)
    return _get_sign_bit(S ^ P)


def gt(x, y):
    """Returns x > y from binary-shared Arrays `x` and `y`."""
    x, y = _flip_sign_bit(x), _flip_sign_bit(y)
    S = x & ~y
    P = ~(x ^ y)
    S, _ = _spk_circuit(S, P)
    return _get_sign_bit(S)


def ge(x, y):
    """Returns x >= y from binary-shared Arrays `x` and `y`."""
    x, y = _flip_sign_bit(x), _flip_sign_bit(y)
    S = x & ~y
    P = ~(x ^ y)
    S, P = _spk_circuit(S, P)
    return _get_sign_bit(S ^ P)
