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
    return p0_share ^ a_share, s1p1_share ^ b_share


@jax.jit
def _spk_combine_and_post(sp_share, c_share, eps, delta, a_share, b_share, out_mask, is_rank0):
    b_eff = jnp.where(is_rank0, b_share ^ delta, b_share)
    update_share = c_share ^ (b_eff & eps) ^ (a_share & delta)
    not_out_mask = out_mask ^ jnp.int64(-1)
    sp_share = sp_share.at[1].set(sp_share[1] & not_out_mask)
    return sp_share ^ update_share


def _spk_circuit(S, P):
    """
    Computes the Set-Propagate-Kill Tree circuit for a set (S, P)
    (K is implied by S, P since (SPK) is one-hot).

    See section 6.3 of Damgard et al., "Unconditionally Secure Constant-Rounds
    Multi-Party Computation for Equality, Comparison, Bits and Exponentiation".
    """
    import prix.communicator as comm
    from prix.array import Array
    from prix.provider import get_default_provider

    # Vectorize private AND calls to reduce rounds:
    SP = Array.stack([S, P])
    masks, out_masks, multipliers = _spk_circuit_constants()
    provider = get_default_provider()
    s_shape = S.size()
    sp_shape = SP.size()
    is_rank0 = comm.get().get_rank() == 0

    for i in range(__LOG_BITS):
        a, b, c = provider.generate_binary_triple(s_shape, sp_shape, device=S.device)
        eps_share, delta_share = _spk_pre_and_mask(
            SP.share, a.share, b.share, masks[i], out_masks[i], multipliers[i]
        )
        eps, delta = comm.get().all_reduce([eps_share, delta_share], op="bxor", batched=True)
        SP.share = _spk_combine_and_post(
            SP.share, c.share, eps, delta, a.share, b.share, out_masks[i], is_rank0
        )

    return SP[0], SP[1]


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
