#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import jax.numpy as jnp

from prix.encoder import encode, decode, DATATYPE


def get_test_array(max_value=10, floating=False):
    """Create simple test array."""

    data = list(range(max_value))
    if floating:
        array = jnp.array(data, dtype=jnp.float32)
    else:
        array = jnp.array(data, dtype=DATATYPE)
    return array


class TestCommon(unittest.TestCase):
    """
    Test cases for common functionality.
    """

    def _check(self, array, reference, msg):
        test_passed = jnp.all(array == reference).item() == 1
        self.assertTrue(test_passed, msg=msg)

    def test_encode_decode(self) -> None:
        """Tests array encoding and decoding."""
        for floating in [False, True]:
            if floating:
                precision = 16
            else:
                precision = 0
            array = get_test_array(floating=floating)
            decoded = decode(encode(array, precision), precision)
            self._check(
                decoded,
                array,
                "Encoding/decoding a %s failed." % "float" if float else "long",
            )

        # Try a few other types.
        precision = 0
        for dtype in [jnp.uint8, jnp.int8, jnp.int16]:
            array = jnp.array([12, 42, 32, 96, 48], dtype=dtype)
            decoded = decode(encode(array, precision), precision).astype(dtype)
            self._check(decoded, array, "Encoding/decoding a %s failed." % dtype)


if __name__ == "__main__":
    unittest.main()
