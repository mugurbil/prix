#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from jax import numpy as jnp

PRECISION = 16
DATATYPE = jnp.int32


def encode(x, precision, device=None):
    """Helper function to wrap data if needed"""

    scale = 2**precision
    if isinstance(x, int) or isinstance(x, float):
        # Squeeze in order to get a 0-dim array with value `x`
        return jnp.array([scale * x], dtype=DATATYPE, device=device).squeeze()
    elif isinstance(x, list):
        return jnp.array(
            jnp.array(x, dtype=jnp.float32, device=device) * scale,
            dtype=DATATYPE,
            device=device,
        )
    elif jnp.issubdtype(x.dtype, jnp.floating):
        return jnp.array(scale * x, dtype=DATATYPE, device=device)
    # For integer types cast to long prior to scaling to avoid overflow.
    elif jnp.issubdtype(x.dtype, jnp.integer):
        return scale * jnp.array(x, dtype=DATATYPE, device=device)
    elif isinstance(x, np.ndarray):
        return scale * jnp.array(x, dtype=DATATYPE, device=device)
    else:
        raise TypeError("Unknown array type: %s." % type(x))


def decode(x, precision):
    """Helper function that decodes from scaled array"""
    if x is None:
        return None

    assert jnp.issubdtype(x.dtype, DATATYPE), "input must be a int array"

    if precision > 0:
        scale = 2**precision
        correction = jnp.array(x < 0, dtype=DATATYPE, device=x.device)
        dividend = jnp.floor(x / (scale - correction))
        remainder = x % scale
        remainder += (remainder == 0).astype(DATATYPE) * scale * correction

        x = dividend.astype(jnp.float32) + remainder.astype(jnp.float32) / scale
    return x
