#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import jax
import numpy as np
from jax import numpy as jnp

from prix.config import cfg

jax.config.update("jax_enable_x64", True)

PRECISION = 16
DATATYPE = jnp.int64


@functools.lru_cache(maxsize=None)
def _default_device():
    return jax.devices()[0]


@functools.lru_cache(maxsize=64)
def resolve_device(device):
    """Resolve a device specification (None, str, or jax.Device) to a jax.Device or None."""
    if device is None:
        return None
    if isinstance(device, jax.Device):
        return device
    if isinstance(device, str):
        dev_str = device.lower()
        if ":" in dev_str:
            backend, idx = dev_str.split(":", 1)
            if backend == "cuda":
                backend = "gpu"
            return jax.devices(backend)[int(idx)]
        if dev_str == "cuda":
            dev_str = "gpu"
        return jax.devices(dev_str)[0]
    return device


def to_jax(arr, device=None, dtype=DATATYPE):
    """Convert an array to a JAX array on `device` without redundant device_put calls."""
    if isinstance(arr, jnp.ndarray):
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        if device is not None:
            dev = resolve_device(device)
            if dev is not None and dev != _default_device():
                arr = jax.device_put(arr, dev)
        return arr
    if device is not None:
        dev = resolve_device(device)
        if dev is not None and dev != _default_device():
            return jax.device_put(np.asarray(arr, dtype=dtype), dev)
    return jnp.asarray(arr, dtype=dtype)


def encode(x, precision=None, device=None):
    """Helper function to encode plaintext data into fixed-point integer ring."""
    if precision is None:
        precision = cfg.encoder.precision_bits

    # Already an encoded/encrypted Prix Array
    if hasattr(x, "share"):
        return x

    scale = int(2**precision)
    if isinstance(x, (int, float, np.number)):
        return to_jax(np.round(scale * x), device=device, dtype=DATATYPE)
    elif isinstance(x, (list, tuple)):
        return to_jax(np.round(np.asarray(x, dtype=np.float64) * scale), device=device, dtype=DATATYPE)
    elif isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.floating):
            return to_jax(np.round(x.astype(np.float64) * scale), device=device, dtype=DATATYPE)
        else:
            return to_jax(x.astype(np.int64) * scale, device=device, dtype=DATATYPE)
    elif hasattr(x, "dtype"):
        if x.dtype == DATATYPE and precision == 0:
            arr = x
        elif jnp.issubdtype(x.dtype, jnp.floating):
            arr = jnp.round(x.astype(jnp.float64) * scale).astype(DATATYPE)
        elif jnp.issubdtype(x.dtype, jnp.integer) or x.dtype == jnp.bool_:
            arr = x.astype(DATATYPE) if scale == 1 else x.astype(DATATYPE) * scale
        else:
            raise TypeError(f"Unsupported array dtype: {x.dtype}.")
        return to_jax(arr, device=device, dtype=DATATYPE)
    else:
        raise TypeError(f"Unknown array type: {type(x)}.")


def decode(x, precision=None):
    """Helper function that decodes from scaled fixed-point integer array."""
    if x is None:
        return None
    if precision is None:
        precision = cfg.encoder.precision_bits

    assert jnp.issubdtype(x.dtype, jnp.integer), f"input must be an int array, got {x.dtype}"
    x = x.astype(DATATYPE)

    if precision > 0:
        scale = int(2**precision)
        dividend = x >> precision
        remainder = x & (scale - 1)
        return dividend.astype(jnp.float32) + remainder.astype(jnp.float32) / float(scale)
    return x


class FixedPointEncoder:
    """Encoder that encodes float/int arrays into scaled 64-bit fixed-point integers."""

    def __init__(self, precision_bits=None):
        if precision_bits is None:
            precision_bits = cfg.encoder.precision_bits
        self._precision_bits = int(precision_bits)

    @property
    def precision_bits(self):
        return self._precision_bits

    @precision_bits.setter
    def precision_bits(self, value):
        self._precision_bits = int(value)

    @property
    def scale(self):
        return int(2**self._precision_bits)

    @property
    def _scale(self):
        return int(2**self._precision_bits)

    @_scale.setter
    def _scale(self, value):
        self._precision_bits = int(np.log2(value))

    def encode(self, x, device=None):
        return encode(x, precision=self._precision_bits, device=device)

    def decode(self, tensor):
        return decode(tensor, precision=self._precision_bits)
