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

import jax
import jax.numpy as jnp
import numpy as np
try:
    from scipy.special import erf as _scipy_erf
except ImportError:
    _scipy_erf = None

import prix
import prix.communicator as comm
from prix.config import cfg
from prix.encoder import DATATYPE, resolve_device, to_jax
from prix.evaluator import EvaluatorClient, check_supported as check_evaluator_supported
from prix.random import PRNGGenerator, _invert_permutation, chebyshev_series, fourier_series

__all__ = [
    "LookupTables",
    "permute_reveal_evaluate_share",
    "_fourier_series",
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


from prix.luts import LookupTables  # noqa: E402  (tables live in prix/luts.py)

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


def _eval_haar_lut(x, lut_table, truncation):
    """Evaluates a Haar Wavelet LUT using Wave's 2-round FSS protocol (`lut_indexing in ('wave', 'wave_prob')`), SIGMA's 3-round TR + DPF-LUT protocol (`lut_indexing == 'sigma'`), or Curl's protocol."""
    if comm.get().get_world_size() == 2 and cfg.encoder.lut_indexing in ("wave", "wave_prob"):
        from prix import fss

        return fss.wave_haar_lut(x, lut_table, truncation=truncation)
    if comm.get().get_world_size() == 2 and cfg.encoder.lut_indexing == "sigma":
        from prix import fss

        msb = fss.sigma_tr_trunc(x, truncation) if truncation > 0 else x
        return fss.wave_haar_lut(msb, lut_table, truncation=0)
    if cfg.encoder.lut_indexing == "crypten":
        msb = x.div(2 ** truncation)
    else:
        msb = x.egk_trunc_pr(62, truncation)
    return msb.evaluate_lut(lut_table)


def _eval_bior_lut(x, lut_table, truncation):
    """Evaluates a Bior(5,3) Wavelet LUT using Wave's 2-round FSS protocol (`lut_indexing in ('wave', 'wave_prob')`), 3-round SIGMA TR + Wave Bior protocol (`lut_indexing == 'sigma'`), or Curl's protocol."""
    if comm.get().get_world_size() == 2 and cfg.encoder.lut_indexing in ("wave", "wave_prob", "sigma"):
        from prix import fss

        return fss.wave_bior_lut(x, lut_table, truncation=truncation)
    if cfg.encoder.lut_indexing == "crypten":
        msb, lsb = x.divmod(2 ** truncation)
    else:
        msb, lsb = x.egk_truncmod_pr(62, truncation)
    return msb.evaluate_bior_lut(lut_table, lsb, truncation)


def _nexp_lut(self, method):
    """Approximates the negative exponential function exp(-x) for x >= 0 using LUTs."""
    luts = LookupTables(self.device)
    precision = 2 ** cfg.encoder.precision_bits
    size = cfg.functions.exp.neg_lut_size

    if method == "split":
        size = int(size)
        step = builtins.max(1, int((precision * 16) // (size * size)))
        step_bits = int(round(math.log2(step))) if step > 1 else 0
        size_bits = int(round(math.log2(size)))
        if cfg.encoder.lut_indexing == "crypten":
            x = self.div(step) if step > 1 else self.clone()
        elif cfg.encoder.lut_indexing in ("sigma", "wave", "wave_prob"):
            if step_bits > 0 and cfg.encoder.lut_indexing == "sigma":
                _ = comm.get().all_reduce(self.share)
            x = self.egk_trunc_pr(62, step_bits) if step_bits > 0 else self.clone()
        else:
            x = self.egk_trunc_pr(62, step_bits) if step_bits > 0 else self.clone()
        d = self < 16
        c = d * x + (1 - d) * (float(size * size - 1) / float(precision))
        t0 = _eval_haar_lut(c, luts.get_lut("nexp_low", self.device), 0)
        t1 = _eval_haar_lut(c, luts.get_lut("nexp_high", self.device), size_bits)
        return t0 * t1
    elif method == "haar":
        check = self < (2 ** cfg.functions.exp.lut_max_bits)
        truncation = (
            cfg.functions.exp.lut_max_bits
            + cfg.encoder.precision_bits
            - cfg.functions.exp.haar_size_bits
        )
        lut = _eval_haar_lut(self, luts.get_lut("nexp_haar", self.device), truncation)
        return check * lut
    elif method == "bior":
        check = self < (2 ** cfg.functions.exp.lut_max_bits)
        truncation = (
            cfg.functions.exp.lut_max_bits
            + cfg.encoder.precision_bits
            - cfg.functions.exp.bior_size_bits
        )
        lut = _eval_bior_lut(self, luts.get_lut("nexp_bior", self.device), truncation)
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
            method = "haar"
        luts = LookupTables(self.device)
        shifted = self + (2 ** cfg.functions.exp.lut_max_bits)
        if method == "haar":
            truncation = (
                cfg.functions.exp.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.exp.haar_size_bits
            )
            return _eval_haar_lut(shifted, luts.get_lut("exp_haar", self.device), truncation)
        elif method == "bior":
            truncation = (
                cfg.functions.exp.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.exp.bior_size_bits
            )
            return _eval_bior_lut(shifted, luts.get_lut("exp_bior", self.device), truncation)
    elif method == "limit":
        iters = cfg.functions.exp.iterations
        result = 1 + self.div(2 ** iters)
        for _ in range(iters):
            result = result.square()
        return result
    elif method == "poly":
        neg_coeffs = [0.90939454, 0.35179902, 0.06433797, 0.00451738]
        neg_c0 = 0.98636898
        if cfg.functions.exp.all_neg:
            do_poly = (-5.0 - self)._ltz()
            poly_neg = self.polynomial(neg_coeffs) + neg_c0
            return do_poly * poly_neg
        pos_coeffs = [0.87648823, 0.82507053, -0.14632820, 0.15607787]
        pos_c0 = 1.01072671
        ltz_pair = prix.stack([self, -5.0 - self])._ltz()
        is_neg, gt_neg5 = ltz_pair[0], ltz_pair[1]
        in_pos = 1 - is_neg
        in_neg = is_neg + gt_neg5 - 1
        trunc_method = cfg.encoder.truncation
        if trunc_method not in ("crypten", "egk"):
            # The fused combination below truncates by hand; other truncation schemes (e.g.
            # Fission's deferred truncation) go through the generic polynomial path.
            poly_pos = self.polynomial(pos_coeffs) + pos_c0
            poly_neg = self.polynomial(neg_coeffs) + neg_c0
            return (prix.stack([in_pos, in_neg]) * prix.stack([poly_pos, poly_neg])).sum(0)
        x2 = self.square()
        x1_x2 = prix.stack([self, x2])
        terms = prix.cat([x1_x2, x1_x2 * x2])
        bcast_shape = (2, 4) + (1,) * self.dim()
        coeffs_enc = prix.encoder.encode(
            jnp.reshape(jnp.asarray([pos_coeffs, neg_coeffs], dtype=jnp.float32), bcast_shape),
            precision=terms.precision,
            device=terms.device,
        )
        both = self.shallow_copy()
        both.share = jnp.sum(jnp.expand_dims(terms.share, axis=0) * coeffs_enc, axis=1)
        both.precision = terms.precision
        if both.precision > 0:
            if trunc_method == "crypten":
                both = both.div_(2 ** both.precision)
            else:
                both = both.egk_trunc_pr(62, both.precision)
        c0_bcast = jnp.reshape(jnp.asarray([pos_c0, neg_c0], dtype=jnp.float32), (2,) + (1,) * self.dim())
        both = both + c0_bcast
        return (prix.stack([in_pos, in_neg]) * both).sum(0)
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
            return _eval_haar_lut(self, luts.get_lut("log_haar", self.device), log_truncation)
        elif method == "bior":
            log_truncation = (
                cfg.functions.log.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.log.bior_size_bits
            )
            return _eval_bior_lut(self, luts.get_lut("log_bior", self.device), log_truncation)
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
            return _eval_haar_lut(
                self, luts.get_lut("reciprocal_haar", self.device), reciprocal_truncation
            )
        elif method == "bior":
            reciprocal_truncation = (
                cfg.functions.reciprocal.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.reciprocal.bior_size_bits
            )
            return _eval_bior_lut(
                self, luts.get_lut("reciprocal_bior", self.device), reciprocal_truncation
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

    if method in ("haar", "bior", "bior_scaled", "tailored_haar"):
        luts = LookupTables(self.device)
        if method == "haar":
            truncation = (
                cfg.functions.inv_sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.inv_sqrt.haar_size_bits
            )
            return _eval_haar_lut(self, luts.get_lut("inv_sqrt_haar", self.device), truncation)
        elif method == "bior":
            truncation = (
                cfg.functions.inv_sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.inv_sqrt.bior_size_bits
            )
            return _eval_bior_lut(self, luts.get_lut("inv_sqrt_bior", self.device), truncation)
        elif method == "bior_scaled":
            # Range-extended 1/sqrt(x) from ONE bior(5,3) table T(v) over [0, 2^lut_max_bits):
            #   x^{-1/2} = 2^{e/2} * T(2^e x)   for any public e,
            # so we evaluate the same table (one batched LUT call, identical (J, j) for all copies)
            # on x * 2^e for e in `range_scales` (descending), each copy covering a 2^e-shifted
            # window of x, and select obliviously with FSS comparisons against the public
            # thresholds 2^(lut_max_bits - e_k) that separate consecutive windows:
            #   y = y_{e_0} + sum_k [x < t_k] * (y_{e_{k+1}} - y_{e_k}),  t_k = 2^(lut_max_bits - e_{k+1}).
            # Scaling a share by a public power of two is free; the selector bits have precision 0,
            # so the mux is a single Beaver round with no truncation. Cost: 1 batched LUT on
            # len(scales) copies + 1 batched comparison (2 rounds) + 1 mux round.
            scales = [int(e) for e in getattr(cfg.functions.inv_sqrt, "range_scales", [16, 8, 0])]
            scales = sorted(set(scales), reverse=True)
            max_bits = int(cfg.functions.inv_sqrt.lut_max_bits)
            truncation = max_bits + cfg.encoder.precision_bits - cfg.functions.inv_sqrt.bior_size_bits
            def _shl(arr, k):  # multiply a share by the public power of two 2^k: free, no truncation
                return arr.from_shares(arr.share * (1 << k), precision=arr.precision, device=arr.device) if k else arr

            if len(scales) == 1:
                e = scales[0]
                return _eval_bior_lut(_shl(self, e), luts.get_lut(f"inv_sqrt_bior_s{e}", self.device), truncation)
            # Ascending order of e means descending x-windows: e_0 = 0 covers the largest x.
            asc = sorted(scales)  # e.g. [0, 8, 16]
            xs = prix.stack([_shl(self, e) for e in asc])
            tables = jnp.stack([luts.get_lut(f"inv_sqrt_bior_s{e}", self.device) for e in asc])  # (K, 2, size)
            ys = _eval_bior_lut(xs, tables, truncation)  # one batched LUT call, one table per window
            thresholds = [float(2 ** (max_bits - asc[k + 1])) for k in range(len(asc) - 1)]
            lt = prix.stack([self - t for t in thresholds]) < 0  # [x < t_k], precision 0
            diffs = prix.stack([ys[k + 1] - ys[k] for k in range(len(asc) - 1)])
            return ys[0] + (lt * diffs).sum(0)
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
            y_0 = _eval_haar_lut(
                self, luts.get_lut("inv_sqrt_tailored_haar_0", self.device), truncation_0
            )
            y_1 = _eval_haar_lut(
                self, luts.get_lut("inv_sqrt_tailored_haar_1", self.device), truncation_1
            )
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
            return _eval_haar_lut(self, luts.get_lut("sqrt_haar", self.device), truncation)
        elif method == "bior":
            truncation = (
                cfg.functions.sqrt.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sqrt.bior_size_bits
            )
            return _eval_bior_lut(self, luts.get_lut("sqrt_bior", self.device), truncation)
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
            cos_val = _eval_haar_lut(x, luts.get_lut("cos_haar", self.device), trig_truncation)
            sin_val = _eval_haar_lut(x, luts.get_lut("sin_haar", self.device), trig_truncation)
        elif method == "bior":
            trig_truncation = cfg.encoder.precision_bits - cfg.functions.trigonometry.bior_size_bits
            cos_val = _eval_bior_lut(x, luts.get_lut("cos_bior", self.device), trig_truncation)
            sin_val = _eval_bior_lut(x, luts.get_lut("sin_bior", self.device), trig_truncation)
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
            cos_val = _eval_haar_lut(x, luts.get_lut("cos_haar_lut_only", self.device), truncation)
            sin_val = _eval_haar_lut(x, luts.get_lut("sin_haar_lut_only", self.device), truncation)
            return cos_val, sin_val
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.trigonometry.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.trigonometry.bior_size_bits
            )
            cos_val = _eval_bior_lut(x, luts.get_lut("cos_bior_lut_only", self.device), truncation)
            sin_val = _eval_bior_lut(x, luts.get_lut("sin_bior_lut_only", self.device), truncation)
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


def _ltz_and_abs_lt(self, width, return_neg_part=False):
    """Computes (self < 0, |self| < width) in a single batched _ltz() call."""
    if comm.get().get_world_size() == 2 and getattr(cfg.mpc, "protocol", "beaver") == "fss":
        from prix import fss

        return fss.fss_ltz_and_abs_lt(self, width, return_neg_part=return_neg_part)
    ltz_all = prix.stack([self, self - width, -self - width])._ltz()
    ltz, check = ltz_all[0], ltz_all[1] + ltz_all[2] - 1
    if return_neg_part:
        return ltz, check, self * ltz
    return ltz, check


def sigmoid(self):
    """Computes the sigmoid function."""
    method = cfg.functions.sigmoid_tanh.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        ltz, check, neg_part = _ltz_and_abs_lt(
            self, 2 ** cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits - 1, return_neg_part=True
        )
        sgn = 1 - 2 * ltz
        abs_val = self - 2 * neg_part
        if method == "haar":
            st_truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.haar_size_bits
            )
            lut = _eval_haar_lut(abs_val, luts.get_lut("sigmoid_haar", self.device), st_truncation)
        elif method == "bior":
            st_truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            lut = _eval_bior_lut(abs_val, luts.get_lut("sigmoid_bior", self.device), st_truncation)
        eval_val = ltz + sgn * lut
        limit = 1 - ltz
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
            return _eval_haar_lut(x, luts.get_lut("sigmoid_haar_lut_only", self.device), truncation)
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.sigmoid_tanh.sigmoid_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            return _eval_bior_lut(x, luts.get_lut("sigmoid_bior_lut_only", self.device), truncation)
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
    elif method == "poly":
        return tanh(self.div(2)).div(2) + 0.5
    elif method == "fourier":
        period = getattr(cfg.functions.sigmoid_tanh, "sigmoid_fs_period", 16)
        width = period / 2.0
        terms = getattr(cfg.functions.sigmoid_tanh, "sigmoid_fs_terms", 8)
        alpha, _, beta_sin = _cached_fourier_coeffs("sigmoid", float(width), int(terms))
        return _fourier_series(self, terms, period, alpha=alpha, beta_sin=beta_sin)
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "sigmoid")
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")


def _erf_np(x):
    if _scipy_erf is not None:
        return _scipy_erf(x)
    return np.asarray(jax.scipy.special.erf(jnp.asarray(x, dtype=jnp.float64)))


def _diff_gelu_np(x):
    gelu_x = 0.5 * x * (1.0 + _erf_np(x / math.sqrt(2.0)))
    relu_x = np.maximum(x, 0.0)
    return np.sign(x) * (gelu_x - relu_x)


def _diff_silu_np(x):
    silu_x = x / (1.0 + np.exp(-x))
    relu_x = np.maximum(x, 0.0)
    return np.sign(x) * (silu_x - relu_x)


@functools.lru_cache(maxsize=64)
def _cached_fourier_coeffs(func_name, width, terms):
    if func_name == "gelu" and width == 4.0 and terms == 8:
        return (
            0.0,
            np.zeros(8, dtype=np.float32),
            np.array(
                [-0.0818, -0.0809, -0.0424, -0.0176, -0.0079, -0.0043, -0.0026, -0.0017],
                dtype=np.float32,
            ),
        )
    if func_name == "silu" and width == 8.0 and terms == 12:
        return (
            0.0,
            np.zeros(12, dtype=np.float32),
            np.array(
                [
                    -0.1299,
                    -0.1220,
                    -0.0743,
                    -0.0394,
                    -0.0216,
                    -0.0118,
                    -0.0074,
                    -0.0044,
                    -0.0033,
                    -0.0021,
                    -0.0018,
                    -0.0011,
                ],
                dtype=np.float32,
            ),
        )
    fn_map = {
        "erf": _erf_np,
        "tanh": np.tanh,
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "gelu": _diff_gelu_np,
        "silu": _diff_silu_np,
    }
    return fourier_series(fn_map[func_name], width, terms)


@jax.jit
def _fourier_linear_comb_jit(u_share, v_share, delta_int, period_scaled, inv_scale_f, k_arr, beta_sin_scaled):
    delta = jnp.mod(delta_int, period_scaled).astype(jnp.float32) * inv_scale_f
    angles = k_arr * jnp.expand_dims(delta, axis=0)
    wv = jnp.round(jnp.sin(angles) * beta_sin_scaled).astype(DATATYPE)
    wu = jnp.round(jnp.cos(angles) * beta_sin_scaled).astype(DATATYPE)
    return jnp.sum(v_share * wv + u_share * wu, axis=0)


@jax.jit
def _fourier_linear_comb_cos_jit(
    u_share, v_share, delta_int, period_scaled, inv_scale_f, k_arr, beta_sin_scaled, beta_cos_scaled
):
    delta = jnp.mod(delta_int, period_scaled).astype(jnp.float32) * inv_scale_f
    angles = k_arr * jnp.expand_dims(delta, axis=0)
    s = jnp.sin(angles)
    c = jnp.cos(angles)
    wv = jnp.round(s * beta_sin_scaled + c * beta_cos_scaled).astype(DATATYPE)
    wu = jnp.round(c * beta_sin_scaled - s * beta_cos_scaled).astype(DATATYPE)
    return jnp.sum(v_share * wv + u_share * wu, axis=0)


def _fourier_series(self, terms, period, alpha=None, beta_cos=None, beta_sin=None):
    """Evaluates a Fourier series at secret-shared tensor `self` in 1 round using trigonometric triples."""
    if beta_sin is None and beta_cos is None:
        raise ValueError("At least one of beta_sin or beta_cos must be provided")

    x = self
    if x.precision > cfg.encoder.precision_bits:
        x = x.egk_trunc_pr(62, x.precision - cfg.encoder.precision_bits)
        x.precision = cfg.encoder.precision_bits

    terms_int = int(terms)
    period_f = float(period)
    scale_f = float(1 << x.precision)
    inv_scale_f = jnp.float32(1.0 / scale_f)
    period_scaled = jnp.int64(int(round(period_f * scale_f)))
    provider = prix.get_default_provider()
    t, u, v = provider.generate_trig_triple(x.size(), period_f, terms_int, device=x.device)

    delta_int = to_jax(comm.get().all_reduce(x.share - t.share), device=x.device, dtype=DATATYPE)
    bcast_shape = (terms_int,) + (1,) * x.dim()
    k_arr = to_jax(
        np.arange(1, terms_int + 1, dtype=np.float32).reshape(bcast_shape)
        * np.float32(2.0 * math.pi / period_f),
        device=x.device,
        dtype=jnp.float32,
    )
    if beta_sin is not None:
        beta_sin_scaled = to_jax(
            np.asarray(beta_sin, dtype=np.float32).reshape(bcast_shape) * scale_f,
            device=x.device,
            dtype=jnp.float32,
        )
    else:
        beta_sin_scaled = jnp.zeros(bcast_shape, dtype=jnp.float32)

    if beta_cos is None:
        sum_share = _fourier_linear_comb_jit(
            u.share, v.share, delta_int, period_scaled, inv_scale_f, k_arr, beta_sin_scaled
        )
    else:
        beta_cos_scaled = to_jax(
            np.asarray(beta_cos, dtype=np.float32).reshape(bcast_shape) * scale_f,
            device=x.device,
            dtype=jnp.float32,
        )
        sum_share = _fourier_linear_comb_cos_jit(
            u.share, v.share, delta_int, period_scaled, inv_scale_f, k_arr, beta_sin_scaled, beta_cos_scaled
        )

    res = x.shallow_copy()
    res.share = sum_share
    res.precision = x.precision
    if res.precision > 0:
        trunc_method = cfg.encoder.truncation
        if trunc_method == "crypten":
            res = res.div_(2 ** res.precision)
        elif trunc_method == "egk":
            res = res.egk_trunc_pr(62, res.precision)
        elif trunc_method == "fission":
            res.precision = 2 * res.precision
        else:
            raise ValueError(f"Unsupported truncation method {trunc_method}")

    if alpha is not None and float(alpha) != 0.0:
        res = res + float(alpha)
    return res


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
        ltz, check, neg_part = _ltz_and_abs_lt(
            self, 2 ** cfg.functions.sigmoid_tanh.tanh_lut_max_bits - 1, return_neg_part=True
        )
        sgn = 1 - 2 * ltz
        abs_val = self - 2 * neg_part
        if method == "haar":
            st_truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.haar_size_bits
            )
            lut = _eval_haar_lut(abs_val, luts.get_lut("tanh_haar", self.device), st_truncation)
        elif method == "bior":
            st_truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            lut = _eval_bior_lut(abs_val, luts.get_lut("tanh_bior", self.device), st_truncation)
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
            return _eval_haar_lut(x, luts.get_lut("tanh_haar_lut_only", self.device), truncation)
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.sigmoid_tanh.tanh_lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.sigmoid_tanh.bior_size_bits
            )
            return _eval_bior_lut(x, luts.get_lut("tanh_bior_lut_only", self.device), truncation)
    elif method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh.terms
        coeffs = chebyshev_series(np.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        out = tanh_polys_flipped.matmul(coeffs)
        return out.hardtanh()
    elif method == "poly":
        ltz, do_poly, neg_part = _ltz_and_abs_lt(self, 2.95, return_neg_part=True)
        sgn = 1 - 2 * ltz
        abs_x = self - 2 * neg_part
        poly_x = abs_x.polynomial([1.1950192, -0.49313435, 0.0737858, -0.00147019]) - 0.01758266
        return sgn * (do_poly * (poly_x - 1) + 1)
    elif method == "fourier":
        period = getattr(cfg.functions.sigmoid_tanh, "tanh_fs_period", 16)
        width = period / 2.0
        terms = getattr(cfg.functions.sigmoid_tanh, "tanh_fs_terms", 8)
        _, _, beta_sin = _cached_fourier_coeffs("tanh", float(width), int(terms))
        return _fourier_series(self, terms, period, beta_sin=beta_sin)
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
            lut = _eval_haar_lut(abs_val, luts.get_lut("erf_haar", self.device), erf_truncation)
        elif method == "bior":
            erf_truncation = (
                cfg.functions.erf.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.erf.bior_size_bits
            )
            lut = _eval_bior_lut(abs_val, luts.get_lut("erf_bior", self.device), erf_truncation)
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
            return _eval_haar_lut(x, luts.get_lut("erf_haar_lut_only", self.device), truncation)
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.erf.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.erf.bior_size_bits
            )
            return _eval_bior_lut(x, luts.get_lut("erf_bior_lut_only", self.device), truncation)
    elif method in ("Taylor", "taylor"):
        iters = cfg.functions.erf.iterations
        output = self.clone()
        for n in range(1, iters + 1):
            multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
            output = output.add(self.pos_pow(2 * n + 1).mul(multiplier))
        return output.mul(2.0 / math.sqrt(math.pi))
    elif method == "tanh":
        return tanh(math.sqrt(4.0 / math.pi) * (self + 0.044715 * self.pow(3)))
    elif method == "poly":
        ltz, do_poly, neg_part = _ltz_and_abs_lt(self, 2.5, return_neg_part=True)
        sgn = 1 - 2 * ltz
        abs_x = self - 2 * neg_part
        poly_x = abs_x.polynomial([1.30561301, -0.45803194, -0.01699669, 0.02278162]) - 0.01501806
        return sgn * (do_poly * (poly_x - 1) + 1)
    elif method == "fourier":
        period = getattr(cfg.functions.erf, "fs_period", 16)
        width = period / 2.0
        terms = getattr(cfg.functions.erf, "fs_terms", 5)
        _, _, beta_sin = _cached_fourier_coeffs("erf", float(width), int(terms))
        return _fourier_series(self, terms, period, beta_sin=beta_sin)
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "erf")
    else:
        raise ValueError(f"Unrecognized method {method} for erf")


def gelu(self):
    """Approximates the GELU function of the input tensor."""
    method = cfg.functions.gelu.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        ltz, check, neg_part = _ltz_and_abs_lt(
            self, 2 ** cfg.functions.gelu.lut_max_bits, return_neg_part=True
        )
        abs_val = self - 2 * neg_part
        relu_val = self - neg_part
        if method == "haar":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.haar_size_bits
            )
            lut = _eval_haar_lut(abs_val, luts.get_lut("gelu_haar", self.device), truncation)
        elif method == "bior":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.bior_size_bits
            )
            lut = _eval_bior_lut(abs_val, luts.get_lut("gelu_bior", self.device), truncation)
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
            return _eval_haar_lut(x, luts.get_lut("gelu_haar_lut_only", self.device), truncation)
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.gelu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.gelu.bior_size_bits
            )
            return _eval_bior_lut(x, luts.get_lut("gelu_bior_lut_only", self.device), truncation)
    elif method == "erf":
        return self * (1 + (self / math.sqrt(2)).erf()) / 2
    elif method == "poly":
        ltz, do_poly, neg_part = _ltz_and_abs_lt(self, 3, return_neg_part=True)
        abs_x = self - 2 * neg_part
        relu_x = self - neg_part
        poly_x = abs_x.polynomial([-0.55386347, 0.5658561, -0.19719836, 0.02328962]) + 0.00410626
        return relu_x + do_poly * poly_x
    elif method == "fourier":
        period = getattr(cfg.functions.gelu, "fs_period", 8)
        width = period / 2.0
        terms = getattr(cfg.functions.gelu, "fs_terms", 8)
        ltz, do_fs, neg_part = _ltz_and_abs_lt(self, width, return_neg_part=True)
        abs_x = self - 2 * neg_part
        relu_x = self - neg_part
        _, _, beta_sin = _cached_fourier_coeffs("gelu", float(width), int(terms))
        return relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
    elif method == "fission":
        return permute_reveal_evaluate_share(self, "gelu")
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")


def silu(self):
    """Approximates the SiLU (Swish) function of the input tensor."""
    method = cfg.functions.silu.method

    if method in ("haar", "bior"):
        luts = LookupTables(self.device)
        ltz, check, neg_part = _ltz_and_abs_lt(
            self, 2 ** cfg.functions.silu.lut_max_bits - 1, return_neg_part=True
        )
        abs_val = self - 2 * neg_part
        relu_val = self - neg_part
        if method == "haar":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.haar_size_bits
            )
            lut = _eval_haar_lut(abs_val, luts.get_lut("silu_haar", self.device), truncation)
        elif method == "bior":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.bior_size_bits
            )
            lut = _eval_bior_lut(abs_val, luts.get_lut("silu_bior", self.device), truncation)
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
            return _eval_haar_lut(x, luts.get_lut("silu_haar_lut_only", self.device), truncation)
        elif method == "bior-lut-only":
            truncation = (
                cfg.functions.silu.lut_max_bits
                + cfg.encoder.precision_bits
                - cfg.functions.silu.bior_size_bits
            )
            return _eval_bior_lut(x, luts.get_lut("silu_bior_lut_only", self.device), truncation)
    elif method == "sigmoid":
        return self * self.sigmoid()
    elif method == "poly":
        ltz, do_poly, neg_part = _ltz_and_abs_lt(self, 6, return_neg_part=True)
        abs_x = self - 2 * neg_part
        relu_x = self - neg_part
        poly_x = abs_x.polynomial([-0.47465521, 0.27018027, -0.05269808, 0.00348313]) - 0.00951912
        return relu_x + do_poly * poly_x
    elif method == "fourier":
        period = getattr(cfg.functions.silu, "fs_period", 16)
        width = period / 2.0
        terms = getattr(cfg.functions.silu, "fs_terms", 12)
        ltz, do_fs, neg_part = _ltz_and_abs_lt(self, width, return_neg_part=True)
        abs_x = self - 2 * neg_part
        relu_x = self - neg_part
        _, _, beta_sin = _cached_fourier_coeffs("silu", float(width), int(terms))
        return relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
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
        if getattr(cfg.mpc, "protocol", "beaver") == "fss" or cfg.functions.max.method == "tree":
            maximum_value = self.max(dim, keepdim=True, return_argmax=False)
        else:
            maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        with cfg.temp_override({"functions.exp.all_neg": True}):
            numerator = logits.exp()
        # The denominator is a sum of L terms in (0, 1], so it lies in [1, L]. Widen the
        # reciprocal LUT domain [0, 2^lut_max_bits) to cover L (SIGMA Sec. 5.2.3 sizes the
        # inverse table from the softmax length), enlarging the table by the same factor so
        # the knot spacing -- and hence the accuracy near 1 -- is unchanged. The reciprocal
        # is only evaluated on the L-times-smaller reduced tensor, so the cost is negligible.
        seq_len = int(self.size(dim))
        need_bits = int(math.ceil(math.log2(seq_len + 1)))
        rec_override = {"functions.reciprocal.all_pos": True}
        extra = need_bits - int(cfg.functions.reciprocal.lut_max_bits)
        if extra > 0:
            rec_override["functions.reciprocal.lut_max_bits"] = need_bits
            rec_override["functions.reciprocal.bior_size_bits"] = int(cfg.functions.reciprocal.bior_size_bits) + extra
            rec_override["functions.reciprocal.haar_size_bits"] = int(cfg.functions.reciprocal.haar_size_bits) + extra
        with cfg.temp_override(rec_override):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
    elif method == "ode":
        iter_num = int(getattr(cfg.functions.softmax, "ode_iter_num", 16))
        clip = bool(getattr(cfg.functions.softmax, "ode_clip", True))
        upper = float(getattr(cfg.functions.softmax, "ode_ub", 12))
        lower = float(getattr(cfg.functions.softmax, "ode_lb", -4))

        logits = self
        if clip:
            diff = prix.stack([logits - upper, lower - logits]).relu()
            logits = logits + diff[1] - diff[0]

        x = logits.div(iter_num)
        g = self.new(
            jnp.full(self.shape, 1.0 / self.size(dim), dtype=jnp.float32),
            device=self.device,
        )
        for _ in range(iter_num):
            g = g + (x - g.mul(x).sum(dim=dim, keepdim=True)) * g
        return g
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
        if getattr(cfg.mpc, "protocol", "beaver") == "fss" or cfg.functions.max.method == "tree":
            maximum_value = self.max(dim, keepdim=True, return_argmax=False)
        else:
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
    if comm.get().get_world_size() == 2 and getattr(cfg.mpc, "protocol", "beaver") == "fss":
        from prix import fss

        return fss.fss_relu(self)
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


def max(self, dim=None, keepdim=False, one_hot=True, return_argmax=True):
    """Returns the maximum value of all elements in the input tensor."""
    method = cfg.functions.max.method
    if dim is None:
        if method in ["tree", "log_reduction", "double_log_reduction"]:
            max_result = _max_helper_all_tree_reductions(self, method=method)
        else:
            with cfg.temp_override({"functions.max.method": method}):
                argmax_result = self.argmax(one_hot=True)
            max_result = self.mul(argmax_result).sum()
        return max_result
    else:
        if not return_argmax and method in [
            "tree",
            "log_reduction",
            "double_log_reduction",
            "accelerated_cascade",
        ]:
            max_result = _max_helper_all_tree_reductions(self, dim=dim, method=method)
            if keepdim:
                max_result = (
                    max_result.unsqueeze(dim) if max_result.dim() < self.dim() else max_result
                )
            return max_result
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
    use_fss = (
        comm.get().get_world_size() == 2
        and getattr(cfg.mpc, "protocol", "beaver") == "fss"
    )
    if use_fss:
        from prix import fss
    enc_tensor_reduced = input_tensor.clone()
    for _ in range(steps):
        m = enc_tensor_reduced.size(dim)
        if m <= 1:
            break
        x, y, remainder = enc_tensor_reduced.split([m // 2, m // 2, m % 2], dim=dim)
        if use_fss:
            pairwise_max = fss.fss_max_pairwise(x, y)
        else:
            pairwise_max = prix.where(x >= y, x, y)
        enc_tensor_reduced = prix.cat([pairwise_max, remainder], dim=dim)
    return enc_tensor_reduced


def _max_helper_tree_reduction(enc_tensor, dim=None):
    if enc_tensor.dim() == 0:
        return enc_tensor
    inp, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        inp = enc_tensor.flatten()
    n = inp.size(dim_used)
    steps = int(math.ceil(math.log2(builtins.max(n, 1))))
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(inp, dim_used, steps)
    return enc_tensor_reduced.squeeze(dim_used)


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
    if method == "tree":
        return _max_helper_tree_reduction(enc_tensor, dim)
    elif method == "log_reduction":
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
    elif method in ["tree", "log_reduction", "double_log_reduction", "accelerated_cascade"]:
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
        return self.square().pow((p - 1) // 2).mul(self)


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
    if not isinstance(coeffs, prix.Array) and cfg.encoder.truncation != "fission":
        bcast_shape = (coeffs.shape[0],) + (1,) * (terms.dim() - 1)
        coeffs_enc = prix.encoder.encode(
            jnp.reshape(jnp.asarray(coeffs, dtype=jnp.float32), bcast_shape),
            precision=terms.precision,
            device=terms.device,
        )
        res = self.shallow_copy()
        res.share = jnp.sum(terms.share * coeffs_enc, axis=0)
        res.precision = terms.precision
        if res.precision > 0:
            trunc_method = cfg.encoder.truncation
            if trunc_method == "crypten":
                res = res.div_(2 ** res.precision)
            elif trunc_method == "egk":
                res = res.egk_trunc_pr(62, res.precision)
        return res

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
    pad_shift = 24 + builtins.max(0, builtins.min(int(self.precision), 30))
    max_input.share, output_size = _pool2d_reshape(
        self.share,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        ceil_mode=ceil_mode,
        pad_value=-(1 << pad_shift),
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
