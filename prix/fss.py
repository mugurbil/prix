#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Function Secret Sharing (FSS) primitives for 2-party MPC in PriX, following
SIGMA (PETS 2024, "SIGMA: Secure GPT Inference with Function Secret Sharing"):

1. Distributed Point Function (`dpf_gen_jit`) and 1-word-per-level prefix
   comparison (`dpf_eval_lt_jit`) computing binary shares of `1{z < alpha}`
   without extra per-level value correction words via the divergence-bit
   identity along the evaluation path.
2. Signed 2's complement MSB / DReLU extraction (`fss_msb_eval_jit`) and
   single-key 3-way shifted interval comparison (`fss_msb_eval_3way_jit`)
   evaluating `(x < 0, |x| < width)` from a single opened `x_hat = x + r_in`
   and a single DPF key.
3. 64-to-1 bit-packed binary reveal (`reveal_packed_bits`) reducing single-bit
   B2A communication by 64x (to 1 bit = 0.125 B per element).
4. 2-round fused DReLU + multiplication (`fss_relu`, `fss_max_pairwise`, and
   `fss_ltz_and_abs_lt`) computing `b * x` in the same round as the 1-bit B2A
   reveal using the preprocessed product `w = rA * r_in`.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np

import prix.communicator as comm
from prix.config import cfg
from prix.encoder import resolve_device, to_jax
from prix.random import _sm64_raw

DATATYPE = jnp.int64

_BIT_SHIFTS_U64 = jnp.arange(64, dtype=jnp.uint64)


@jax.jit
def _pack_bits_1d_jit(padded_u64):
    """Packs a 1D uint64 array of length M (multiple of 64) with {0,1} values into M//64 uint64s."""
    blocks = (padded_u64 & jnp.uint64(1)).reshape(-1, 64)
    shifted = blocks << _BIT_SHIFTS_U64[None, :]
    packed = jnp.bitwise_or.reduce(shifted, axis=1)
    return jax.lax.bitcast_convert_type(packed, DATATYPE)


@functools.partial(jax.jit, static_argnames=("n_orig",))
def _unpack_bits_1d_jit(packed_i64, n_orig: int):
    """Unpacks a 1D int64 array of packed bits into n_orig int64 values in {0, 1}."""
    packed_u64 = jax.lax.bitcast_convert_type(packed_i64, jnp.uint64)
    bits_2d = (packed_u64[:, None] >> _BIT_SHIFTS_U64[None, :]) & jnp.uint64(1)
    return bits_2d.reshape(-1)[:n_orig].astype(DATATYPE)


def pack_bits(bit_share):
    """Packs an arbitrary-shape {0, 1} int64 JAX array into a 64x smaller 1D int64 array."""
    flat = jnp.asarray(bit_share, dtype=DATATYPE).reshape(-1)
    n = flat.shape[0]
    if n == 0:
        return flat
    rem = n % 64
    if rem != 0:
        flat = jnp.pad(flat, (0, 64 - rem))
    return _pack_bits_1d_jit(jax.lax.bitcast_convert_type(flat, jnp.uint64))


def unpack_bits(packed_i64, shape):
    """Unpacks a 1D packed int64 array back to `shape` with values in {0, 1}."""
    n = int(np.prod(shape)) if len(shape) > 0 else 1
    if n == 0:
        return jnp.zeros(shape, dtype=DATATYPE)
    flat = _unpack_bits_1d_jit(packed_i64, n)
    return flat.reshape(shape)


def reveal_packed_bits(bit_share):
    """Reveals a binary-shared {0, 1} tensor using 64-to-1 bit-packed all_reduce(op='bxor')."""
    shape = bit_share.shape
    packed = pack_bits(bit_share)
    revealed_packed = comm.get().all_reduce(packed, op="bxor")
    return unpack_bits(revealed_packed, shape)


@jax.jit
def _prg2_u64(seeds: jnp.ndarray):
    """
    Vectorized 2-output PRG mapping each 62-bit seed in `seeds` (uint64) to
    `(sL, tL, sR, tR)` using two rounds of SplitMix64 mixing.
    Clears bits 0 and 1 of `sL, sR` so each level's correction word
    `(s_cw, t_cw_L, t_cw_R)` packs into a single 64-bit word.
    """
    z1 = seeds + jnp.uint64(0x9E3779B97F4A7C15)
    z1 = (z1 ^ (z1 >> jnp.uint64(30))) * jnp.uint64(0xBF58476D1CE4E5B9)
    z1 = (z1 ^ (z1 >> jnp.uint64(27))) * jnp.uint64(0x94D049BB133111EB)
    z1 = z1 ^ (z1 >> jnp.uint64(31))

    z2 = seeds + jnp.uint64(0x3C6EF372FE94F82A)
    z2 = (z2 ^ (z2 >> jnp.uint64(30))) * jnp.uint64(0xBF58476D1CE4E5B9)
    z2 = (z2 ^ (z2 >> jnp.uint64(27))) * jnp.uint64(0x94D049BB133111EB)
    z2 = z2 ^ (z2 >> jnp.uint64(31))

    tL = z1 & jnp.uint64(1)
    tR = z2 & jnp.uint64(1)
    mask = ~jnp.uint64(3)
    sL = z1 & mask
    sR = z2 & mask
    return sL, tL, sR, tR


@functools.partial(jax.jit, static_argnames=("nbits",))
def dpf_gen_jit(alpha: jnp.ndarray, k0_0: jnp.ndarray, k1_0: jnp.ndarray, nbits: int = 63):
    """
    Vectorized DPF key generation for prefix comparison `< alpha` on `nbits`-bit unsigned inputs.
    Returns `cw_all` of shape `(nbits, N)` uint64 where each level packs
    `s_cw | t_cw_L | (t_cw_R << 1)` into a single 64-bit word.
    """
    mask = ~jnp.uint64(3)
    s0 = k0_0 & mask
    s1 = k1_0 & mask
    t0 = jnp.zeros_like(s0)
    t1 = jnp.ones_like(s1)

    def body(i, carry):
        s0, t0, s1, t1, cw_all = carry
        shift = jnp.uint64(nbits - 1 - i)
        alpha_i = (alpha >> shift) & jnp.uint64(1)
        s0L, t0L, s0R, t0R = _prg2_u64(s0)
        s1L, t1L, s1R, t1R = _prg2_u64(s1)

        go_right = alpha_i == jnp.uint64(1)
        s_lose_0 = jnp.where(go_right, s0L, s0R)
        s_lose_1 = jnp.where(go_right, s1L, s1R)
        s_cw = s_lose_0 ^ s_lose_1

        t_cw_L = t0L ^ t1L ^ alpha_i ^ jnp.uint64(1)
        t_cw_R = t0R ^ t1R ^ alpha_i

        cw_packed = s_cw | t_cw_L | (t_cw_R << jnp.uint64(1))
        cw_all = cw_all.at[i].set(cw_packed)

        s0_keep = jnp.where(go_right, s0R, s0L)
        s1_keep = jnp.where(go_right, s1R, s1L)
        t0_keep = jnp.where(go_right, t0R, t0L)
        t1_keep = jnp.where(go_right, t1R, t1L)
        t_cw_keep = jnp.where(go_right, t_cw_R, t_cw_L)

        s0_next = s0_keep ^ jnp.where(t0 == jnp.uint64(1), s_cw, jnp.uint64(0))
        s1_next = s1_keep ^ jnp.where(t1 == jnp.uint64(1), s_cw, jnp.uint64(0))
        t0_next = t0_keep ^ (t0 & t_cw_keep)
        t1_next = t1_keep ^ (t1 & t_cw_keep)
        return s0_next, t0_next, s1_next, t1_next, cw_all

    cw_init = jnp.zeros((nbits,) + alpha.shape, dtype=jnp.uint64)
    _, _, _, _, cw_all = jax.lax.fori_loop(0, nbits, body, (s0, t0, s1, t1, cw_init))
    return cw_all


@functools.partial(jax.jit, static_argnames=("party_id", "nbits"))
def dpf_eval_lt_jit(
    x: jnp.ndarray,
    s_init: jnp.ndarray,
    cw_all: jnp.ndarray,
    party_id: int,
    nbits: int = 63,
):
    """
    Vectorized DPF prefix-comparison evaluation returning a binary (XOR) share
    in `{0, 1}` of `1{x < alpha}` for `nbits`-bit unsigned integer `x`.
    """
    mask = ~jnp.uint64(3)
    s = s_init & mask
    t = jnp.full_like(s, jnp.uint64(party_id))
    acc = jnp.zeros_like(s)

    def body(i, carry):
        s, t, acc = carry
        shift = jnp.uint64(nbits - 1 - i)
        x_i = (x >> shift) & jnp.uint64(1)
        sL, tL, sR, tR = _prg2_u64(s)

        cw_packed = cw_all[i]
        s_cw = cw_packed & mask
        t_cw_L = cw_packed & jnp.uint64(1)
        t_cw_R = (cw_packed >> jnp.uint64(1)) & jnp.uint64(1)

        go_right = x_i == jnp.uint64(1)
        s_next_raw = jnp.where(go_right, sR, sL)
        t_next_raw = jnp.where(go_right, tR, tL)
        t_cw = jnp.where(go_right, t_cw_R, t_cw_L)

        s_next = s_next_raw ^ jnp.where(t == jnp.uint64(1), s_cw, jnp.uint64(0))
        t_next = t_next_raw ^ (t & t_cw)

        acc = acc ^ ((jnp.uint64(1) - x_i) & (t ^ t_next))
        return s_next, t_next, acc

    _, _, acc = jax.lax.fori_loop(0, nbits, body, (s, t, acc))
    return acc


@functools.partial(jax.jit, static_argnames=("party_id", "nbits"))
def fss_msb_eval_jit(
    x_hat_i64: jnp.ndarray,
    s_init: jnp.ndarray,
    cw_all: jnp.ndarray,
    msb_r_sh: jnp.ndarray,
    party_id: int,
    nbits: int = 63,
):
    """
    Evaluates the signed 2's complement MSB / DReLU (`x < 0`) from opened
    `x_hat = x + r_in` using SIGMA's Lemma 1 identity:
      MSB_n(x) = MSB_n(x_hat) ^ MSB_n(-r_in) ^ 1{ (~x_hat mod 2^(n-1)) < (-r_in mod 2^(n-1)) }
    """
    x_hat_u = jax.lax.bitcast_convert_type(x_hat_i64.reshape(-1), jnp.uint64)
    low_mask = (
        (jnp.uint64(1) << jnp.uint64(nbits)) - jnp.uint64(1)
        if nbits < 64
        else ~jnp.uint64(0)
    )
    z = (~x_hat_u) & low_mask
    lt_sh = dpf_eval_lt_jit(z, s_init, cw_all, party_id, nbits)
    msb_xhat = (x_hat_u >> jnp.uint64(nbits)) & jnp.uint64(1)
    out_sh = lt_sh ^ msb_r_sh
    if party_id == 1:
        out_sh = out_sh ^ msb_xhat
    return out_sh.astype(DATATYPE).reshape(x_hat_i64.shape)


@functools.partial(jax.jit, static_argnames=("party_id", "nbits"))
def fss_msb_eval_3way_jit(
    x_hat_i64: jnp.ndarray,
    w_int: jnp.ndarray,
    s_init: jnp.ndarray,
    cw_all: jnp.ndarray,
    msb_r_sh: jnp.ndarray,
    party_id: int,
    nbits: int = 63,
):
    """
    Evaluates 3 shifted comparisons on the SAME DPF key and the SAME opened `x_hat = x + r_in`:
      1. `x < 0`           at `x_hat`
      2. `x - w < 0`       at `x_hat - w`
      3. `x + w - 1 < 0`   at `x_hat + w - 1` (which is `x <= -w`)
    Returns binary shares `(ltz_bin, check_bin)` of `(1{x < 0}, 1{|x| < w})` stacked
    along axis 0 with shape `(2, *x_hat.shape)`.
    """
    orig_shape = x_hat_i64.shape
    xh0 = jax.lax.bitcast_convert_type(x_hat_i64.reshape(-1), jnp.uint64)
    w_u = jax.lax.bitcast_convert_type(jnp.int64(w_int), jnp.uint64)
    xh1 = xh0 - w_u
    xh2 = xh0 + w_u - jnp.uint64(1)

    low_mask = (
        (jnp.uint64(1) << jnp.uint64(nbits)) - jnp.uint64(1)
        if nbits < 64
        else ~jnp.uint64(0)
    )
    z0 = (~xh0) & low_mask
    z1 = (~xh1) & low_mask
    z2 = (~xh2) & low_mask

    # Evaluate all 3 queries in a single fused loop over the tree levels
    mask = ~jnp.uint64(3)
    s0 = s_init & mask
    s1 = s0
    s2 = s0
    t0 = jnp.full_like(s0, jnp.uint64(party_id))
    t1 = t0
    t2 = t0
    acc0 = jnp.zeros_like(s0)
    acc1 = acc0
    acc2 = acc0

    def body(i, carry):
        s0, t0, acc0, s1, t1, acc1, s2, t2, acc2 = carry
        shift = jnp.uint64(nbits - 1 - i)
        cw_packed = cw_all[i]
        s_cw = cw_packed & mask
        t_cw_L = cw_packed & jnp.uint64(1)
        t_cw_R = (cw_packed >> jnp.uint64(1)) & jnp.uint64(1)

        def step_one(z, s, t, acc):
            z_i = (z >> shift) & jnp.uint64(1)
            sL, tL, sR, tR = _prg2_u64(s)
            go_r = z_i == jnp.uint64(1)
            s_nr = jnp.where(go_r, sR, sL)
            t_nr = jnp.where(go_r, tR, tL)
            t_cw = jnp.where(go_r, t_cw_R, t_cw_L)
            s_n = s_nr ^ jnp.where(t == jnp.uint64(1), s_cw, jnp.uint64(0))
            t_n = t_nr ^ (t & t_cw)
            acc_n = acc ^ ((jnp.uint64(1) - z_i) & (t ^ t_n))
            return s_n, t_n, acc_n

        s0, t0, acc0 = step_one(z0, s0, t0, acc0)
        s1, t1, acc1 = step_one(z1, s1, t1, acc1)
        s2, t2, acc2 = step_one(z2, s2, t2, acc2)
        return s0, t0, acc0, s1, t1, acc1, s2, t2, acc2

    _, _, acc0, _, _, acc1, _, _, acc2 = jax.lax.fori_loop(
        0, nbits, body, (s0, t0, acc0, s1, t1, acc1, s2, t2, acc2)
    )

    b0 = acc0 ^ msb_r_sh
    # Notice: in b1 ^ b2, msb_r_sh XORs with itself and cancels out!
    b_check = acc1 ^ acc2
    if party_id == 1:
        msb0 = (xh0 >> jnp.uint64(nbits)) & jnp.uint64(1)
        msb1 = (xh1 >> jnp.uint64(nbits)) & jnp.uint64(1)
        msb2 = (xh2 >> jnp.uint64(nbits)) & jnp.uint64(1)
        b0 = b0 ^ msb0
        b_check = b_check ^ msb1 ^ msb2

    return jnp.stack(
        [b0.astype(DATATYPE).reshape(orig_shape), b_check.astype(DATATYPE).reshape(orig_shape)],
        axis=0,
    )


@functools.partial(jax.jit, static_argnames=("is_rank0",))
def _b2a_combine_jit(rA_share, z, is_rank0: bool):
    res = rA_share * (jnp.int64(1) - jnp.int64(2) * z)
    return (res + z) if is_rank0 else res


@jax.jit
def _drelu_mul_combine_jit(x_share, x_hat, rA_share, w_share, z):
    """
    Given opened `x_hat = x + r_in` (Round 1) and opened `z = b ^ rB` (Round 2),
    computes additive shares of `relu(x) = x - b * x` in 0 extra rounds:
      (b * x)_i = z * x_i + (1 - 2*z) * (rA_i * x_hat - w_i)
    where `w = rA * r_in (mod 2^64)`.
    """
    bx_share = z * x_share + (jnp.int64(1) - jnp.int64(2) * z) * (rA_share * x_hat - w_share)
    return x_share - bx_share


@functools.partial(jax.jit, static_argnames=("is_rank0",))
def _ltz_abs_lt_combine_jit(x_share, x_hat, rA_share, w_share, z_both, is_rank0: bool):
    """
    Reconstructs arithmetic shares of `(ltz, check, neg_part = ltz * x)` from the
    single 2-bit-per-element B2A reveal `z_both = [z_ltz, z_check]`.
    """
    z0, z1 = z_both[0], z_both[1]
    rA0, rA1 = rA_share[0], rA_share[1]
    sign0 = jnp.int64(1) - jnp.int64(2) * z0
    sign1 = jnp.int64(1) - jnp.int64(2) * z1
    ltz_sh = rA0 * sign0 + jnp.where(is_rank0, z0, jnp.int64(0))
    check_sh = rA1 * sign1 + jnp.where(is_rank0, z1, jnp.int64(0))
    neg_part_sh = z0 * x_share + sign0 * (rA0 * x_hat - w_share)
    return ltz_sh, check_sh, neg_part_sh


@functools.partial(
    jax.jit,
    static_argnames=("shape", "nbits", "mode", "is_rank0"),
)
def _tfp_dcf_key_jit(
    c_glob, c0, c1, s_glob, s0, s1, shape, nbits: int, mode: str, is_rank0: bool
):
    """
    Vectorized, JIT-compiled TFP dealer generating correlated FSS/DCF keys
    and B2A / DReLU-mul masks without inter-party communication.
    """
    n = int(np.prod(shape)) if len(shape) > 0 else 1
    glob_raw = _sm64_raw(s_glob, c_glob, (4, n))
    r_in = glob_raw[0]
    k0_0 = jax.lax.bitcast_convert_type(glob_raw[1], jnp.uint64) & ~jnp.uint64(3)
    k1_0 = jax.lax.bitcast_convert_type(glob_raw[2], jnp.uint64) & ~jnp.uint64(3)
    r_bits = glob_raw[3]

    neg_r_u = jax.lax.bitcast_convert_type(-r_in, jnp.uint64)
    low_mask = (
        (jnp.uint64(1) << jnp.uint64(nbits)) - jnp.uint64(1)
        if nbits < 64
        else ~jnp.uint64(0)
    )
    alpha = neg_r_u & low_mask
    msb_r = (neg_r_u >> jnp.uint64(nbits)) & jnp.uint64(1)
    cw_all = dpf_gen_jit(alpha, k0_0, k1_0, nbits)
    s_init = k0_0 if is_rank0 else k1_0

    if mode == "ltz":
        p0 = _sm64_raw(s0, c0, (4, n))
        p1 = _sm64_raw(s1, c1, (4, n))
        diff_a = p0[:2] - p1[:2]
        diff_b = (p0[2:] ^ p1[2:]) & jnp.int64(1)
        r0 = r_bits & jnp.int64(1)
        if is_rank0:
            r_in_sh = (diff_a[0] + r_in).reshape(shape)
            rA_sh = (diff_a[1] + r0).reshape(shape)
            msb_r_sh = jax.lax.bitcast_convert_type(diff_b[0], jnp.uint64) ^ msb_r
            rB_sh = (diff_b[1] ^ r0).reshape(shape)
        else:
            r_in_sh = diff_a[0].reshape(shape)
            rA_sh = diff_a[1].reshape(shape)
            msb_r_sh = jax.lax.bitcast_convert_type(diff_b[0], jnp.uint64)
            rB_sh = diff_b[1].reshape(shape)
        return r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh
    elif mode == "relu":
        p0 = _sm64_raw(s0, c0, (5, n))
        p1 = _sm64_raw(s1, c1, (5, n))
        diff_a = p0[:3] - p1[:3]
        diff_b = (p0[3:] ^ p1[3:]) & jnp.int64(1)
        r0 = r_bits & jnp.int64(1)
        w = r0 * r_in
        if is_rank0:
            r_in_sh = (diff_a[0] + r_in).reshape(shape)
            rA_sh = (diff_a[1] + r0).reshape(shape)
            w_sh = (diff_a[2] + w).reshape(shape)
            msb_r_sh = jax.lax.bitcast_convert_type(diff_b[0], jnp.uint64) ^ msb_r
            rB_sh = (diff_b[1] ^ r0).reshape(shape)
        else:
            r_in_sh = diff_a[0].reshape(shape)
            rA_sh = diff_a[1].reshape(shape)
            w_sh = diff_a[2].reshape(shape)
            msb_r_sh = jax.lax.bitcast_convert_type(diff_b[0], jnp.uint64)
            rB_sh = diff_b[1].reshape(shape)
        return r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh, w_sh
    else:
        p0 = _sm64_raw(s0, c0, (7, n))
        p1 = _sm64_raw(s1, c1, (7, n))
        diff_a = p0[:4] - p1[:4]
        diff_b = (p0[4:] ^ p1[4:]) & jnp.int64(1)
        r0 = r_bits & jnp.int64(1)
        r1 = (r_bits >> jnp.int64(1)) & jnp.int64(1)
        r_pair = jnp.stack([r0, r1], axis=0)
        w0 = r0 * r_in
        if is_rank0:
            r_in_sh = (diff_a[0] + r_in).reshape(shape)
            rA_sh = (diff_a[1:3] + r_pair).reshape((2,) + shape)
            w_sh = (diff_a[3] + w0).reshape(shape)
            msb_r_sh = jax.lax.bitcast_convert_type(diff_b[0], jnp.uint64) ^ msb_r
            rB_sh = (diff_b[1:3] ^ r_pair).reshape((2,) + shape)
        else:
            r_in_sh = diff_a[0].reshape(shape)
            rA_sh = diff_a[1:3].reshape((2,) + shape)
            w_sh = diff_a[3].reshape(shape)
            msb_r_sh = jax.lax.bitcast_convert_type(diff_b[0], jnp.uint64)
            rB_sh = diff_b[1:3].reshape((2,) + shape)
        return r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh, w_sh


def get_fss_nbits():
    """Returns the bit-width `nbits = fss_bits - 1` used for DCF prefix comparisons."""
    fss_bits = int(getattr(cfg.mpc, "fss_bits", 64))
    return max(16, min(63, fss_bits - 1))


def fss_ltz(x):
    """
    2-round FSS comparison `x < 0` (DReLU):
      - Round 1: open `x_hat = x + r_in` (8 B/elem), evaluate DPF locally.
      - Round 2: open bit-packed `z = b ^ rB` (0.125 B/elem), reconstruct `b` in Z_{2^64}.
    """
    from prix.array import Array, Ptype
    from prix.provider import get_default_provider

    provider = get_default_provider()
    nbits = get_fss_nbits()
    party_id = comm.get().get_rank()
    is_rank0 = party_id == 0

    r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh = provider.generate_dcf_key(
        x.size(), nbits=nbits, mode="ltz", device=x.device
    )
    x_hat = comm.get().all_reduce(x.share + r_in_sh, op="sum")
    b_sh = fss_msb_eval_jit(x_hat, s_init, cw_all, msb_r_sh, party_id, nbits)
    z = reveal_packed_bits(b_sh ^ rB_sh)
    out_share = _b2a_combine_jit(rA_sh, z, is_rank0)
    return Array.from_shares(out_share, precision=0, device=x.device, ptype=Ptype.arithmetic)


def fss_relu(x):
    """
    2-round fused FSS ReLU (`x * (x >= 0)`):
      - Round 1: open `x_hat = x + r_in` (8 B/elem), evaluate DPF locally for `b = 1{x < 0}`.
      - Round 2: open bit-packed `z = b ^ rB` (0.125 B/elem), locally compute `x - b * x`
        using preprocessed `w = rA * r_in` (0 extra rounds, 0 Beaver multiplications).
    """
    from prix.array import Array, Ptype
    from prix.provider import get_default_provider

    provider = get_default_provider()
    nbits = get_fss_nbits()
    party_id = comm.get().get_rank()
    is_rank0 = party_id == 0

    r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh, w_sh = provider.generate_dcf_key(
        x.size(), nbits=nbits, mode="relu", device=x.device
    )
    x_hat = comm.get().all_reduce(x.share + r_in_sh, op="sum")
    b_sh = fss_msb_eval_jit(x_hat, s_init, cw_all, msb_r_sh, party_id, nbits)
    z = reveal_packed_bits(b_sh ^ rB_sh)
    out_share = _drelu_mul_combine_jit(x.share, x_hat, rA_sh, w_sh, z)
    return Array.from_shares(
        out_share, precision=x.precision, device=x.device, ptype=Ptype.arithmetic
    )


def fss_max_pairwise(x, y):
    """
    2-round pairwise maximum `max(x, y) = y + ReLU(x - y)` using fused FSS ReLU.
    Costs 2 rounds and 8.125 B/elem (vs. 9 rounds and 232 B/elem in GMW+Beaver).
    """
    return y + fss_relu(x - y)


def fss_ltz_and_abs_lt(x, width, return_neg_part=False):
    """
    2-round multi-comparison computing `(x < 0, |x| < width)` (and optionally
    `neg_part = (x < 0) * x`) using a SINGLE DPF key and a SINGLE opened `x_hat = x + r_in`:
      - Round 1: open `x_hat = x + r_in` (8 B/elem), evaluate DPF at `x_hat`, `x_hat - w`,
        and `x_hat + w - 1`.
      - Round 2: open 2 bit-packed B2A bits (0.25 B/elem), reconstruct `ltz`, `check`,
        and `neg_part = ltz * x`.
    """
    from prix.array import Array, Ptype
    from prix.provider import get_default_provider

    provider = get_default_provider()
    nbits = get_fss_nbits()
    party_id = comm.get().get_rank()
    is_rank0 = party_id == 0

    w_int = int(round(float(width) * (1 << x.precision)))
    r_in_sh, s_init, cw_all, msb_r_sh, rA_sh, rB_sh, w_sh = provider.generate_dcf_key(
        x.size(), nbits=nbits, mode="ltz_abs_lt", device=x.device
    )
    x_hat = comm.get().all_reduce(x.share + r_in_sh, op="sum")
    b_both = fss_msb_eval_3way_jit(x_hat, w_int, s_init, cw_all, msb_r_sh, party_id, nbits)
    z_both = reveal_packed_bits(b_both ^ rB_sh)
    ltz_sh, check_sh, neg_part_sh = _ltz_abs_lt_combine_jit(
        x.share, x_hat, rA_sh, w_sh, z_both, is_rank0
    )
    ltz_arr = Array.from_shares(ltz_sh, precision=0, device=x.device, ptype=Ptype.arithmetic)
    check_arr = Array.from_shares(check_sh, precision=0, device=x.device, ptype=Ptype.arithmetic)
    if return_neg_part:
        neg_part_arr = Array.from_shares(
            neg_part_sh, precision=x.precision, device=x.device, ptype=Ptype.arithmetic
        )
        return ltz_arr, check_arr, neg_part_arr
    return ltz_arr, check_arr


# ==============================================================================
# Wave (PoPETs 2025) Wavelet LUT Protocols using Function Secret Sharing (FSS):
#   - \Pi^{LUT-Haar}      (Figure 10 / Haar.hpp): 2 rounds, 24 B/elem
#   - \Pi^{LUT-bior(5,3)} (Figure 6  / bior.hpp): 2 rounds, 40 B/elem
# ==============================================================================


@jax.jit
def _prg4_u64(seeds: jnp.ndarray):
    """
    Vectorized 4-output PRG mapping each 62-bit seed in `seeds` (uint64) to
    `(sL, tL, vL, sR, tR, vR)` for Boyle et al. arithmetic DCF over Z_{2^64}.
    """
    sL, tL, sR, tR = _prg2_u64(seeds)
    z3 = seeds + jnp.uint64(0x6A09E667F3BCC908)
    z3 = (z3 ^ (z3 >> jnp.uint64(30))) * jnp.uint64(0xBF58476D1CE4E5B9)
    z3 = (z3 ^ (z3 >> jnp.uint64(27))) * jnp.uint64(0x94D049BB133111EB)
    vL = jax.lax.bitcast_convert_type(z3 ^ (z3 >> jnp.uint64(31)), DATATYPE)

    z4 = seeds + jnp.uint64(0xBB67AE8584CAA73B)
    z4 = (z4 ^ (z4 >> jnp.uint64(30))) * jnp.uint64(0xBF58476D1CE4E5B9)
    z4 = (z4 ^ (z4 >> jnp.uint64(27))) * jnp.uint64(0x94D049BB133111EB)
    vR = jax.lax.bitcast_convert_type(z4 ^ (z4 >> jnp.uint64(31)), DATATYPE)
    return sL, tL, vL, sR, tR, vR


@jax.jit
def _convert_g(seeds: jnp.ndarray):
    """Maps a 62-bit seed to an arithmetic group element in Z_{2^64}."""
    z = (seeds & ~jnp.uint64(3)) + jnp.uint64(0x510E527FADE682D1)
    z = (z ^ (z >> jnp.uint64(30))) * jnp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> jnp.uint64(27))) * jnp.uint64(0x94D049BB133111EB)
    return jax.lax.bitcast_convert_type(z ^ (z >> jnp.uint64(31)), DATATYPE)


@functools.partial(jax.jit, static_argnames=("nbits",))
def dcf_gen_arith_jit(alpha: jnp.ndarray, k0_0: jnp.ndarray, k1_0: jnp.ndarray, nbits: int):
    """
    Boyle et al. (2021) arithmetic-output DCF key generation for `< alpha` with payload
    `beta = 1` in Z_{2^64} (`WaveHelloToPrivacy/experiments/include/dcf.hpp`).
    Returns `(cw_all, v_cw_all, g_cw)`.
    """
    mask = ~jnp.uint64(3)
    s0 = k0_0 & mask
    s1 = k1_0 & mask
    t0 = jnp.zeros_like(s0)
    t1 = jnp.ones_like(s1)
    V = jnp.zeros(alpha.shape, dtype=DATATYPE)

    def body(i, carry):
        s0, t0, s1, t1, V, cw_all, v_cw_all = carry
        shift = jnp.uint64(nbits - 1 - i)
        alpha_i = (alpha >> shift) & jnp.uint64(1)
        go_right = alpha_i == jnp.uint64(1)

        s0L, t0L, v0L, s0R, t0R, v0R = _prg4_u64(s0)
        s1L, t1L, v1L, s1R, t1R, v1R = _prg4_u64(s1)

        s_lose_0 = jnp.where(go_right, s0L, s0R)
        s_lose_1 = jnp.where(go_right, s1L, s1R)
        s_cw = s_lose_0 ^ s_lose_1

        t_cw_L = t0L ^ t1L ^ alpha_i ^ jnp.uint64(1)
        t_cw_R = t0R ^ t1R ^ alpha_i
        cw_packed = s_cw | t_cw_L | (t_cw_R << jnp.uint64(1))
        cw_all = cw_all.at[i].set(cw_packed)

        v0_lose = jnp.where(go_right, v0L, v0R)
        v1_lose = jnp.where(go_right, v1L, v1R)
        v0_keep = jnp.where(go_right, v0R, v0L)
        v1_keep = jnp.where(go_right, v1R, v1L)

        sign_t1 = jnp.where(t1 == jnp.uint64(1), jnp.int64(-1), jnp.int64(1))
        target_lose = alpha_i.astype(DATATYPE)
        v_cw = sign_t1 * (target_lose - V - (v0_lose - v1_lose))
        v_cw_all = v_cw_all.at[i].set(v_cw)

        V_next = V + (v0_keep - v1_keep) + sign_t1 * v_cw

        s0_keep = jnp.where(go_right, s0R, s0L)
        s1_keep = jnp.where(go_right, s1R, s1L)
        t0_keep = jnp.where(go_right, t0R, t0L)
        t1_keep = jnp.where(go_right, t1R, t1L)
        t_cw_keep = jnp.where(go_right, t_cw_R, t_cw_L)

        s0_next = s0_keep ^ jnp.where(t0 == jnp.uint64(1), s_cw, jnp.uint64(0))
        s1_next = s1_keep ^ jnp.where(t1 == jnp.uint64(1), s_cw, jnp.uint64(0))
        t0_next = t0_keep ^ (t0 & t_cw_keep)
        t1_next = t1_keep ^ (t1 & t_cw_keep)
        return s0_next, t0_next, s1_next, t1_next, V_next, cw_all, v_cw_all

    cw_init = jnp.zeros((nbits,) + alpha.shape, dtype=jnp.uint64)
    v_cw_init = jnp.zeros((nbits,) + alpha.shape, dtype=DATATYPE)
    _, _, _, t1_end, V_end, cw_all, v_cw_all = jax.lax.fori_loop(
        0, nbits, body, (s0, t0, s1, t1, V, cw_init, v_cw_init)
    )
    sign_t1_end = jnp.where(t1_end == jnp.uint64(1), jnp.int64(-1), jnp.int64(1))
    g_cw = sign_t1_end * (-V_end)
    return cw_all, v_cw_all, g_cw


@functools.partial(jax.jit, static_argnames=("party_id", "nbits"))
def dcf_eval_arith_jit(
    x: jnp.ndarray,
    s_init: jnp.ndarray,
    cw_all: jnp.ndarray,
    v_cw_all: jnp.ndarray,
    g_cw: jnp.ndarray,
    party_id: int,
    nbits: int,
):
    """
    Evaluates the Boyle et al. arithmetic DCF on `x`, returning an additive share in Z_{2^64}
    such that `share_0 + share_1 = 1{x < alpha} (mod 2^64)`.
    """
    mask = ~jnp.uint64(3)
    s = s_init & mask
    t = jnp.full_like(s, jnp.uint64(party_id))
    acc = jnp.zeros(x.shape, dtype=DATATYPE)
    sign = jnp.int64(-1 if party_id == 1 else 1)

    def body(i, carry):
        s, t, acc = carry
        shift = jnp.uint64(nbits - 1 - i)
        x_i = (x >> shift) & jnp.uint64(1)
        go_right = x_i == jnp.uint64(1)

        sL, tL, vL, sR, tR, vR = _prg4_u64(s)
        cw_packed = cw_all[i]
        s_cw = cw_packed & mask
        t_cw_L = cw_packed & jnp.uint64(1)
        t_cw_R = (cw_packed >> jnp.uint64(1)) & jnp.uint64(1)
        v_cw = v_cw_all[i]

        s_next_raw = jnp.where(go_right, sR, sL)
        t_next_raw = jnp.where(go_right, tR, tL)
        v_raw = jnp.where(go_right, vR, vL)
        t_cw = jnp.where(go_right, t_cw_R, t_cw_L)

        acc = acc + sign * (v_raw + jnp.where(t == jnp.uint64(1), v_cw, jnp.int64(0)))
        s_next = s_next_raw ^ jnp.where(t == jnp.uint64(1), s_cw, jnp.uint64(0))
        t_next = t_next_raw ^ (t & t_cw)
        return s_next, t_next, acc

    _, t_end, acc = jax.lax.fori_loop(0, nbits, body, (s, t, acc))
    acc = acc + sign * jnp.where(t_end == jnp.uint64(1), g_cw, jnp.int64(0))
    return acc


@functools.partial(jax.jit, static_argnames=("party_id", "J", "j_depth"))
def grotto_segment_parities_jit(
    x_J: jnp.ndarray,
    z_lo: jnp.ndarray,
    s_init: jnp.ndarray,
    cw_all: jnp.ndarray,
    party_id: int,
    J: int,
    j_depth: int,
):
    """
    Evaluates the `(J + j_depth)`-level DPF tree using Grotto's segment-parity algorithm
    with deterministic truncation (`WaveHelloToPrivacy`, Section 3.3 & 5.1):
      1. Full-domain expansion of the top `J` levels to `2^J` nodes `(s_tree, t_tree)`.
      2. Prefix-parity evaluation along `z_lo = 2^j - 1 - x_lo` on the bottom `j_depth`
         levels producing `beta[m] = 1{m == r_J} * borrow_J` (in XOR sharing).
      3. Borrow-adjusted one-hot parity shares:
           q[m] = t_tree[m] ^ beta[m] ^ beta[(m + 1) mod 2^J]
         shifted cyclically by `x_J` so `p[i] = q[(i + x_J) mod 2^J]` is an exact binary
         share of `1{i == msb_J(a)}`.
    Returns `p` of shape `(2^J, N)` uint64 with values in `{0, 1}`.
    """
    mask = ~jnp.uint64(3)
    s_tree = (s_init & mask)[None, :]
    t_tree = jnp.full_like(s_tree, jnp.uint64(party_id))

    for lvl in range(J):
        sL, tL, sR, tR = _prg2_u64(s_tree)
        cw_packed = cw_all[lvl][None, :]
        s_cw = cw_packed & mask
        t_cw_L = cw_packed & jnp.uint64(1)
        t_cw_R = (cw_packed >> jnp.uint64(1)) & jnp.uint64(1)

        t_is_1 = t_tree == jnp.uint64(1)
        sL_next = sL ^ jnp.where(t_is_1, s_cw, jnp.uint64(0))
        sR_next = sR ^ jnp.where(t_is_1, s_cw, jnp.uint64(0))
        tL_next = tL ^ (t_tree & t_cw_L)
        tR_next = tR ^ (t_tree & t_cw_R)

        s_tree = jnp.stack([sL_next, sR_next], axis=1).reshape(1 << (lvl + 1), -1)
        t_tree = jnp.stack([tL_next, tR_next], axis=1).reshape(1 << (lvl + 1), -1)

    t_top = t_tree
    if j_depth > 0:
        beta = jnp.zeros_like(t_tree)

        def low_body(k, carry):
            s_cur, t_cur, beta_cur = carry
            shift = jnp.uint64(j_depth - 1 - k)
            z_k = ((z_lo >> shift) & jnp.uint64(1))[None, :]
            sL, tL, sR, tR = _prg2_u64(s_cur)
            cw_packed = cw_all[J + k][None, :]
            s_cw = cw_packed & mask
            t_cw_L = cw_packed & jnp.uint64(1)
            t_cw_R = (cw_packed >> jnp.uint64(1)) & jnp.uint64(1)

            go_right = z_k == jnp.uint64(1)
            s_nr = jnp.where(go_right, sR, sL)
            t_nr = jnp.where(go_right, tR, tL)
            t_cw = jnp.where(go_right, t_cw_R, t_cw_L)

            s_next = s_nr ^ jnp.where(t_cur == jnp.uint64(1), s_cw, jnp.uint64(0))
            t_next = t_nr ^ (t_cur & t_cw)
            beta_next = beta_cur ^ ((jnp.uint64(1) - z_k) & (t_cur ^ t_next))
            return s_next, t_next, beta_next

        _, _, beta = jax.lax.fori_loop(0, j_depth, low_body, (s_tree, t_tree, beta))
        q = t_top ^ beta ^ jnp.roll(beta, shift=-1, axis=0)
    else:
        q = t_top

    two_J = 1 << J
    idx = (
        jnp.arange(two_J, dtype=jnp.uint64)[:, None]
        + (x_J[None, :] & jnp.uint64(two_J - 1))
    ) & jnp.uint64(two_J - 1)
    return jnp.take_along_axis(q, idx.astype(jnp.int32), axis=0)


@functools.partial(
    jax.jit,
    static_argnames=("shape", "J", "j_depth", "mode", "is_rank0"),
)
def _tfp_wave_key_jit(
    c_glob,
    c0,
    c1,
    s_glob,
    s0,
    s1,
    shape,
    J: int,
    j_depth: int,
    mode: str,
    is_rank0: bool,
):
    """
    Vectorized, JIT-compiled TFP dealer generating correlated Wave FSS keys:
      - `mode == 'haar'`: (J + j_depth)-level DPF key + Beaver multiplication triple (U, IP, U*IP).
      - `mode == 'bior'`: (J + j_depth)-level DPF key + j_depth-level arithmetic DCF key +
        Generalized Beaver 8-tuple (U, A, B, X, A*X - B, U*X, U*A, U*(A*X - B)).
    """
    n = int(np.prod(shape)) if len(shape) > 0 else 1
    two_J_mask = jnp.uint64((1 << J) - 1)
    two_j_mask = jnp.uint64((1 << j_depth) - 1) if j_depth > 0 else jnp.uint64(0)

    if mode == "haar":
        glob_raw = _sm64_raw(s_glob, c_glob, (5, n))
        r_i64 = glob_raw[0]
        r_u = jax.lax.bitcast_convert_type(r_i64, jnp.uint64)
        k0_dpf = jax.lax.bitcast_convert_type(glob_raw[1], jnp.uint64) & ~jnp.uint64(3)
        k1_dpf = jax.lax.bitcast_convert_type(glob_raw[2], jnp.uint64) & ~jnp.uint64(3)
        U = glob_raw[3]
        IP = glob_raw[4]
        C = U * IP

        r_J = (r_u >> jnp.uint64(j_depth)) & two_J_mask
        if j_depth > 0:
            r_lo = r_u & two_j_mask
            alpha_lo = two_j_mask - r_lo
            alpha_full = (r_J << jnp.uint64(j_depth)) | alpha_lo
        else:
            alpha_full = r_J

        cw_dpf = dpf_gen_jit(alpha_full, k0_dpf, k1_dpf, J + j_depth)
        s_dpf = k0_dpf if is_rank0 else k1_dpf

        p0 = _sm64_raw(s0, c0, (4, n))
        p1 = _sm64_raw(s1, c1, (4, n))
        diff = p0 - p1
        if is_rank0:
            r_sh = diff[0] + r_i64
            bvr_sh = diff[1:4] + jnp.stack([U, IP, C], axis=0)
        else:
            r_sh = diff[0]
            bvr_sh = diff[1:4]
        return r_sh, s_dpf, cw_dpf, bvr_sh
    elif mode == "bior_prob":
        glob_raw = _sm64_raw(s_glob, c_glob, (7, n))
        r_i64 = glob_raw[0]
        r_u = jax.lax.bitcast_convert_type(r_i64, jnp.uint64)
        k0_dpf = jax.lax.bitcast_convert_type(glob_raw[1], jnp.uint64) & ~jnp.uint64(3)
        k1_dpf = jax.lax.bitcast_convert_type(glob_raw[2], jnp.uint64) & ~jnp.uint64(3)
        U = glob_raw[3]
        A = glob_raw[4]
        B = glob_raw[5]
        X = glob_raw[6]
        AXmB = A * X - B
        UX = U * X
        UA = U * A
        UAXmB = U * AXmB

        r_J = r_u & two_J_mask
        cw_dpf = dpf_gen_jit(r_J, k0_dpf, k1_dpf, J)
        s_dpf = k0_dpf if is_rank0 else k1_dpf

        p0 = _sm64_raw(s0, c0, (9, n))
        p1 = _sm64_raw(s1, c1, (9, n))
        diff = p0 - p1
        bvr_plain = jnp.stack([U, A, B, X, AXmB, UX, UA, UAXmB], axis=0)
        if is_rank0:
            r_sh = diff[0] + r_i64
            bvr_sh = diff[1:9] + bvr_plain
        else:
            r_sh = diff[0]
            bvr_sh = diff[1:9]
        return r_sh, s_dpf, cw_dpf, bvr_sh
    else:
        glob_raw = _sm64_raw(s_glob, c_glob, (9, n))
        r_i64 = glob_raw[0]
        r_u = jax.lax.bitcast_convert_type(r_i64, jnp.uint64)
        k0_dpf = jax.lax.bitcast_convert_type(glob_raw[1], jnp.uint64) & ~jnp.uint64(3)
        k1_dpf = jax.lax.bitcast_convert_type(glob_raw[2], jnp.uint64) & ~jnp.uint64(3)
        k0_dcf = jax.lax.bitcast_convert_type(glob_raw[3], jnp.uint64) & ~jnp.uint64(3)
        k1_dcf = jax.lax.bitcast_convert_type(glob_raw[4], jnp.uint64) & ~jnp.uint64(3)
        U = glob_raw[5]
        A = glob_raw[6]
        B = glob_raw[7]
        X = glob_raw[8]
        AXmB = A * X - B
        UX = U * X
        UA = U * A
        UAXmB = U * AXmB

        r_J = (r_u >> jnp.uint64(j_depth)) & two_J_mask
        r_lo = r_u & two_j_mask
        alpha_lo = two_j_mask - r_lo
        alpha_full = (r_J << jnp.uint64(j_depth)) | alpha_lo

        cw_dpf = dpf_gen_jit(alpha_full, k0_dpf, k1_dpf, J + j_depth)
        cw_dcf, v_cw_dcf, g_cw_dcf = dcf_gen_arith_jit(alpha_lo, k0_dcf, k1_dcf, j_depth)
        s_dpf = k0_dpf if is_rank0 else k1_dpf
        s_dcf = k0_dcf if is_rank0 else k1_dcf

        p0 = _sm64_raw(s0, c0, (10, n))
        p1 = _sm64_raw(s1, c1, (10, n))
        diff = p0 - p1
        r_lo_i64 = jax.lax.bitcast_convert_type(r_lo, DATATYPE)
        bvr_plain = jnp.stack([U, A, B, X, AXmB, UX, UA, UAXmB], axis=0)
        if is_rank0:
            r_sh = diff[0] + r_i64
            r_lo_sh = diff[1] + r_lo_i64
            bvr_sh = diff[2:10] + bvr_plain
        else:
            r_sh = diff[0]
            r_lo_sh = diff[1]
            bvr_sh = diff[2:10]
        return r_sh, r_lo_sh, s_dpf, cw_dpf, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf, bvr_sh


@functools.partial(jax.jit, static_argnames=("party_id", "J", "j_depth"))
def _wave_haar_r1_jit(
    x_open: jnp.ndarray,
    s_dpf: jnp.ndarray,
    cw_dpf: jnp.ndarray,
    lut: jnp.ndarray,
    bvr_sh: jnp.ndarray,
    party_id: int,
    J: int,
    j_depth: int,
):
    x_u = jax.lax.bitcast_convert_type(x_open, jnp.uint64)
    two_J_mask = jnp.uint64((1 << J) - 1)
    x_J = (x_u >> jnp.uint64(j_depth)) & two_J_mask
    if j_depth > 0:
        two_j_mask = jnp.uint64((1 << j_depth) - 1)
        z_lo = two_j_mask - (x_u & two_j_mask)
    else:
        z_lo = jnp.zeros_like(x_u)

    p = grotto_segment_parities_jit(x_J, z_lo, s_dpf, cw_dpf, party_id, J, j_depth)
    v = p.astype(DATATYPE) if party_id == 0 else -p.astype(DATATYPE)
    u_sh = jnp.sum(v, axis=0)
    ip_sh = jnp.sum(v * lut[:, None], axis=0)
    return jnp.stack([u_sh + bvr_sh[0], ip_sh + bvr_sh[1]], axis=0)


@functools.partial(jax.jit, static_argnames=("is_rank0",))
def _wave_haar_r2_jit(hat_both: jnp.ndarray, bvr_sh: jnp.ndarray, is_rank0: bool):
    uu, ii = hat_both[0], hat_both[1]
    U_sh, IP_sh, C_sh = bvr_sh[0], bvr_sh[1], bvr_sh[2]
    if is_rank0:
        return uu * (ii - IP_sh) - ii * U_sh + C_sh
    return -uu * IP_sh - ii * U_sh + C_sh


@functools.partial(jax.jit, static_argnames=("party_id", "J", "j_depth"))
def _wave_bior_r1_jit(
    x_open: jnp.ndarray,
    r_lo_sh: jnp.ndarray,
    s_dpf: jnp.ndarray,
    cw_dpf: jnp.ndarray,
    s_dcf: jnp.ndarray,
    cw_dcf: jnp.ndarray,
    v_cw_dcf: jnp.ndarray,
    g_cw_dcf: jnp.ndarray,
    luts: jnp.ndarray,
    bvr_sh: jnp.ndarray,
    party_id: int,
    J: int,
    j_depth: int,
):
    c0_table = luts[0]
    c1_table = jnp.round((luts[1] - luts[0]) / float(1 << j_depth)).astype(DATATYPE)

    x_u = jax.lax.bitcast_convert_type(x_open, jnp.uint64)
    two_J_mask = jnp.uint64((1 << J) - 1)
    two_j_mask = jnp.uint64((1 << j_depth) - 1)
    x_J = (x_u >> jnp.uint64(j_depth)) & two_J_mask
    x_lo = x_u & two_j_mask
    z_lo = two_j_mask - x_lo

    p = grotto_segment_parities_jit(x_J, z_lo, s_dpf, cw_dpf, party_id, J, j_depth)
    v = p.astype(DATATYPE) if party_id == 0 else -p.astype(DATATYPE)
    u_sh = jnp.sum(v, axis=0)
    a_ip_sh = jnp.sum(v * c1_table[:, None], axis=0)
    b_ip_sh = jnp.sum(v * c0_table[:, None], axis=0)

    borrow_sh = dcf_eval_arith_jit(z_lo, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf, party_id, j_depth)
    x_lsb_sh = r_lo_sh + (jnp.int64(1 << j_depth) * borrow_sh)
    if party_id == 0:
        x_lsb_sh = x_lsb_sh - jax.lax.bitcast_convert_type(x_lo, DATATYPE)

    return jnp.stack(
        [
            u_sh + bvr_sh[0],
            a_ip_sh + bvr_sh[1],
            b_ip_sh + bvr_sh[2],
            x_lsb_sh + bvr_sh[3],
        ],
        axis=0,
    )


@functools.partial(jax.jit, static_argnames=("party_id", "J", "j_depth"))
def _wave_bior_prob_r1_jit(
    x_open: jnp.ndarray,
    x_lsb_sh: jnp.ndarray,
    s_dpf: jnp.ndarray,
    cw_dpf: jnp.ndarray,
    luts: jnp.ndarray,
    bvr_sh: jnp.ndarray,
    party_id: int,
    J: int,
    j_depth: int,
):
    c0_table = luts[0]
    c1_table = jnp.round((luts[1] - luts[0]) / float(1 << j_depth)).astype(DATATYPE)

    x_u = jax.lax.bitcast_convert_type(x_open, jnp.uint64)
    two_J_mask = jnp.uint64((1 << J) - 1)
    x_J = x_u & two_J_mask
    z_lo = jnp.zeros_like(x_u)

    p = grotto_segment_parities_jit(x_J, z_lo, s_dpf, cw_dpf, party_id, J, 0)
    v = p.astype(DATATYPE) if party_id == 0 else -p.astype(DATATYPE)
    u_sh = jnp.sum(v, axis=0)
    a_ip_sh = jnp.sum(v * c1_table[:, None], axis=0)
    b_ip_sh = jnp.sum(v * c0_table[:, None], axis=0)

    return jnp.stack(
        [
            u_sh + bvr_sh[0],
            a_ip_sh + bvr_sh[1],
            b_ip_sh + bvr_sh[2],
            x_lsb_sh + bvr_sh[3],
        ],
        axis=0,
    )


@functools.partial(jax.jit, static_argnames=("j_depth", "is_rank0"))
def _wave_bior_r2_jit(hat_4: jnp.ndarray, bvr_sh: jnp.ndarray, j_depth: int, is_rank0: bool):
    uu, aa, bb, xx = hat_4[0], hat_4[1], hat_4[2], hat_4[3]
    U_sh, A_sh, B_sh, X_sh = bvr_sh[0], bvr_sh[1], bvr_sh[2], bvr_sh[3]
    AXmB_sh, UX_sh, UA_sh, UAXmB_sh = bvr_sh[4], bvr_sh[5], bvr_sh[6], bvr_sh[7]
    axb = aa * xx + bb
    if is_rank0:
        y_raw = (
            uu * (axb - aa * X_sh - xx * A_sh + AXmB_sh)
            - axb * U_sh
            + aa * UX_sh
            + xx * UA_sh
            - UAXmB_sh
        )
    else:
        y_raw = (
            uu * (-aa * X_sh - xx * A_sh + AXmB_sh)
            - axb * U_sh
            + aa * UX_sh
            + xx * UA_sh
            - UAXmB_sh
        )
    return y_raw >> jnp.int64(j_depth)


@functools.partial(jax.jit, static_argnames=("shape", "j_depth", "is_rank0"))
def _tfp_sigma_tr_key_jit(c_glob, c0, c1, s_glob, s0, s1, shape, j_depth: int, is_rank0: bool):
    """
    Vectorized, JIT-compiled TFP dealer for SIGMA's 1-round faithful Truncate-Reduce
    (Pi^TR_{n,f}, Theorem 4 / Wave Figure 5): generates mask `r_tr`, its high/low shares,
    and a single-path `j_depth`-level arithmetic DCF key for `1{x_hat_lo < r_lo}`.
    """
    n = int(np.prod(shape)) if len(shape) > 0 else 1
    two_j_mask = jnp.uint64((1 << j_depth) - 1)
    glob_raw = _sm64_raw(s_glob, c_glob, (3, n))
    r_i64 = glob_raw[0]
    r_u = jax.lax.bitcast_convert_type(r_i64, jnp.uint64)
    k0_dcf = jax.lax.bitcast_convert_type(glob_raw[1], jnp.uint64) & ~jnp.uint64(3)
    k1_dcf = jax.lax.bitcast_convert_type(glob_raw[2], jnp.uint64) & ~jnp.uint64(3)

    r_lo = r_u & two_j_mask
    r_hi_i64 = jax.lax.bitcast_convert_type(r_u >> jnp.uint64(j_depth), DATATYPE)
    cw_dcf, v_cw_dcf, g_cw_dcf = dcf_gen_arith_jit(r_lo, k0_dcf, k1_dcf, j_depth)
    s_dcf = k0_dcf if is_rank0 else k1_dcf

    p0 = _sm64_raw(s0, c0, (2, n))
    p1 = _sm64_raw(s1, c1, (2, n))
    diff = p0 - p1
    if is_rank0:
        r_sh = diff[0] + r_i64
        r_hi_sh = diff[1] + r_hi_i64
    else:
        r_sh = diff[0]
        r_hi_sh = diff[1]
    return r_sh, r_hi_sh, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf


@functools.partial(jax.jit, static_argnames=("party_id", "j_depth"))
def _sigma_tr_eval_jit(
    x_hat: jnp.ndarray,
    r_hi_sh: jnp.ndarray,
    s_dcf: jnp.ndarray,
    cw_dcf: jnp.ndarray,
    v_cw_dcf: jnp.ndarray,
    g_cw_dcf: jnp.ndarray,
    party_id: int,
    j_depth: int,
):
    x_hat_u = jax.lax.bitcast_convert_type(x_hat, jnp.uint64)
    two_j_mask = jnp.uint64((1 << j_depth) - 1)
    x_hat_lo = x_hat_u & two_j_mask
    x_hat_hi = jax.lax.bitcast_convert_type(x_hat_u >> jnp.uint64(j_depth), DATATYPE)
    carry_sh = dcf_eval_arith_jit(x_hat_lo, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf, party_id, j_depth)
    if party_id == 0:
        return x_hat_hi - r_hi_sh - carry_sh
    return -r_hi_sh - carry_sh


def sigma_tr_trunc(x, truncation: int):
    """
    SIGMA's 1-round faithful Truncate-Reduce (Pi^TR_{n,f}, Theorem 4):
    truncates `x` by `truncation` bits using 1 online round (8 B/elem) and a single-path
    `truncation`-level arithmetic DCF evaluation (`truncation` PRG calls).
    """
    from prix.array import Array, Ptype

    j_depth = int(truncation)
    if j_depth <= 0:
        return x
    shape = x.size()
    x_flat = x.share.reshape(-1)
    n = int(x_flat.size)
    if n == 0:
        return x
    import prix

    party_id = comm.get().get_rank()
    is_rank0 = party_id == 0
    g_glob = prix.generators.get("global")
    g0 = prix.generators.get("prev")
    g1 = prix.generators.get("next")
    s_glob, c_glob = g_glob.next_offset(3 * n)
    s0, c0 = g0.next_offset(2 * n)
    s1, c1 = g1.next_offset(2 * n)
    from prix.provider import _acct_fss_key

    _acct_fss_key(n, dcf_levels=j_depth, ring_elems=2)  # j-level arithmetic DCF + (r, r_hi) shares
    r_sh, r_hi_sh, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf = _tfp_sigma_tr_key_jit(
        c_glob, c0, c1, s_glob, s0, s1, shape, j_depth, is_rank0
    )
    x_hat = comm.get().all_reduce(x_flat + r_sh, op="sum")
    msb_sh = _sigma_tr_eval_jit(
        x_hat, r_hi_sh, s_dcf, cw_dcf, v_cw_dcf, g_cw_dcf, party_id, j_depth
    )
    return Array.from_shares(
        msb_sh.reshape(shape),
        precision=x.precision,
        device=x.device,
        ptype=Ptype.arithmetic,
    )


def wave_haar_lut(x, lut, truncation=0, probabilistic=None):
    r"""
    2-round Wave Haar LUT evaluation (\Pi^{LUT-Haar}, Figure 10 / `Haar.hpp`):
      - Round 1 (8 B/elem): open `x_open = r - a`, evaluate (J + j)-level Grotto DPF
        segment parities with deterministic truncation by `truncation` bits (or J-level
        DPF if `probabilistic=True` / `lut_indexing == 'wave_prob'`).
      - Round 2 (16 B/elem): Beaver multiplication `[u] * [ip]`.
    Total: 2 rounds, 24 B/elem.
    """
    from prix.array import Array, Ptype
    from prix.provider import get_default_provider

    if probabilistic is None:
        probabilistic = cfg.encoder.lut_indexing == "wave_prob"

    lut_jax = to_jax(lut, device=x.device, dtype=DATATYPE)
    size = int(lut_jax.shape[0])
    J = int(round(np.log2(size)))
    if (1 << J) != size:
        raise ValueError(f"wave_haar_lut requires power-of-2 LUT size, got {size}")
    j_depth = int(truncation)
    shape = x.size()
    x_flat = x.share.reshape(-1)
    if x_flat.size == 0:
        return Array.from_shares(
            jnp.zeros(shape, dtype=DATATYPE),
            precision=cfg.encoder.precision_bits,
            device=x.device,
            ptype=Ptype.arithmetic,
        )

    provider = get_default_provider()
    party_id = comm.get().get_rank()
    is_rank0 = party_id == 0

    if probabilistic and j_depth > 0:
        x_flat = x_flat >> jnp.int64(j_depth)
        if is_rank0:
            x_flat = x_flat + jnp.int64(1)
        j_depth = 0

    r_sh, s_dpf, cw_dpf, bvr_sh = provider.generate_wave_key(
        shape, J=J, j_depth=j_depth, mode="haar", device=x.device
    )
    x_open = comm.get().all_reduce(r_sh - x_flat, op="sum")
    hat_both_sh = _wave_haar_r1_jit(
        x_open, s_dpf, cw_dpf, lut_jax, bvr_sh, party_id, J, j_depth
    )
    hat_both = comm.get().all_reduce(hat_both_sh, op="sum")
    y_sh = _wave_haar_r2_jit(hat_both, bvr_sh, is_rank0)
    return Array.from_shares(
        y_sh.reshape(shape),
        precision=cfg.encoder.precision_bits,
        device=x.device,
        ptype=Ptype.arithmetic,
    )


def wave_bior_lut(x, luts, truncation, probabilistic=None):
    r"""
    2-round Wave Biorthogonal (5,3) LUT evaluation (\Pi^{LUT-bior(5,3)}, Figure 6 / `bior.hpp`):
      - Round 1 (8 B/elem): open `x_open = r - a`, evaluate (J + j)-level Grotto DPF
        segment parities for `([u], [a_ip], [b_ip])` and j-level arithmetic DCF for
        exact `[x_lsb]` (or J-level DPF with local probabilistic truncation if
        `probabilistic=True` / `lut_indexing == 'wave_prob'`).
      - Round 2 (32 B/elem): evaluate `[u * (a_ip * x_lsb + b_ip)]` via Wave's
        generalized Beaver 8-tuple (`beaver.cpp` / Appendix G).
    Total: 2 rounds, 40 B/elem (vs. 4 rounds, 40 B/elem in Curl).
    """
    from prix.array import Array, Ptype
    from prix.provider import get_default_provider

    if probabilistic is None:
        probabilistic = cfg.encoder.lut_indexing == "wave_prob"

    luts_jax = to_jax(luts, device=x.device, dtype=DATATYPE)
    size = int(luts_jax.shape[1])
    J = int(round(np.log2(size)))
    if (1 << J) != size:
        raise ValueError(f"wave_bior_lut requires power-of-2 LUT size, got {size}")
    j_depth = int(truncation)
    if j_depth <= 0:
        raise ValueError(f"wave_bior_lut requires truncation > 0, got {j_depth}")
    shape = x.size()
    x_flat = x.share.reshape(-1)
    if x_flat.size == 0:
        return Array.from_shares(
            jnp.zeros(shape, dtype=DATATYPE),
            precision=cfg.encoder.precision_bits,
            device=x.device,
            ptype=Ptype.arithmetic,
        )

    provider = get_default_provider()
    party_id = comm.get().get_rank()
    is_rank0 = party_id == 0

    if probabilistic or cfg.encoder.lut_indexing == "sigma":
        if cfg.encoder.lut_indexing == "sigma" and not probabilistic:
            x_high_sh = sigma_tr_trunc(x, j_depth).share.reshape(-1)
            x_lsb_sh = x_flat - (x_high_sh << jnp.int64(j_depth))
        else:
            x_high_sh = x_flat >> jnp.int64(j_depth)
            x_lsb_sh = x_flat & jnp.int64((1 << j_depth) - 1)
            if is_rank0:
                x_high_sh = x_high_sh + jnp.int64(1)
                x_lsb_sh = x_lsb_sh - jnp.int64(1 << j_depth)
        r_sh, s_dpf, cw_dpf, bvr_sh = provider.generate_wave_key(
            shape, J=J, j_depth=0, mode="bior_prob", device=x.device
        )
        x_open = comm.get().all_reduce(r_sh - x_high_sh, op="sum")
        hat_4_sh = _wave_bior_prob_r1_jit(
            x_open, x_lsb_sh, s_dpf, cw_dpf, luts_jax, bvr_sh, party_id, J, j_depth
        )
        hat_4 = comm.get().all_reduce(hat_4_sh, op="sum")
        y_sh = _wave_bior_r2_jit(hat_4, bvr_sh, j_depth, is_rank0)
        return Array.from_shares(
            y_sh.reshape(shape),
            precision=cfg.encoder.precision_bits,
            device=x.device,
            ptype=Ptype.arithmetic,
        )

    (
        r_sh,
        r_lo_sh,
        s_dpf,
        cw_dpf,
        s_dcf,
        cw_dcf,
        v_cw_dcf,
        g_cw_dcf,
        bvr_sh,
    ) = provider.generate_wave_key(shape, J=J, j_depth=j_depth, mode="bior", device=x.device)
    x_open = comm.get().all_reduce(r_sh - x_flat, op="sum")
    hat_4_sh = _wave_bior_r1_jit(
        x_open,
        r_lo_sh,
        s_dpf,
        cw_dpf,
        s_dcf,
        cw_dcf,
        v_cw_dcf,
        g_cw_dcf,
        luts_jax,
        bvr_sh,
        party_id,
        J,
        j_depth,
    )
    hat_4 = comm.get().all_reduce(hat_4_sh, op="sum")
    y_sh = _wave_bior_r2_jit(hat_4, bvr_sh, j_depth, is_rank0)
    return Array.from_shares(
        y_sh.reshape(shape),
        precision=cfg.encoder.precision_bits,
        device=x.device,
        ptype=Ptype.arithmetic,
    )

