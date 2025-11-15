#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp

import prix.communicator as comm
from prix.encoder import encode, decode, DATATYPE
from prix.random import random

SENTINEL = -1


class Array:
    """
    Array object that uses additive sharing to perform computations.

    Additive shares are computed by splitting each value of the input array
    into N separate random values that add to the input array, where N is
    the number of parties present in the protocol (world_size).
    """

    def __init__(
        self,
        array=None,
        src: int = None,
        precision: int = None,
        device=None,
    ):
        """
        Creates the shared array from the input `array` provided by party `src`.

        The other parties can specify an `array` or `shape` to determine the shape
        of the shared array object to create. In this case, all parties must
        specify the same array shape to prevent the party's shares from varying
        in shape, which leads to undefined behavior.

        Alternatively, the parties can set `broadcast_size` to `True` to have the
        `src` party broadcast the correct shape. The parties who do not know the
        array shape beforehand can provide an empty array as input. This is
        guaranteed to produce correct behavior but requires an additional
        communication round.

        The parties can also set the `precision` and `device` for their share of
        the array. If `device` is unspecified, it is set to `array.device`.
        """
        # Do nothing if source is sentinel:
        if src == SENTINEL:
            return

        # Assertions on inputs:
        assert (
            isinstance(src, int) and 0 <= src < comm.get().get_world_size()
        ), "specified source party does not exist"
        if comm.get().get_rank() == src:
            assert array is not None, "source must provide a data array"
            if hasattr(array, "src"):
                assert (
                    array.src == src
                ), "source of data array must match source of encryption"

        # If device is unspecified, try and get it from array:
        if device is None and array is not None and hasattr(array, "device"):
            device = array.device

        self.precision = precision
        self.device = device

        if comm.get().get_rank() == src:
            # Encode the array
            if array is not None:
                # if jnp.issubdtype(array.dtype, jnp.integer) and precision != 0:
                #     array = array.float()
                array = encode(array, precision=precision, device=device)
                # array = jax.device_put(array, device)
            array_sum = array.sum()

            # Secret share the array
            for i in range(comm.get().get_world_size()):
                if i == comm.get().get_rank():
                    continue
                share_i = random(array_sum + i, array.shape)
                comm.get().send(share_i, i)
                array = array - share_i
        else:
            array = comm.get().recv(array, src)

        self.share = array

    def reconstruct(self, dst=None):
        """Decrypts the tensor with decoding."""
        return decode(self.reveal(dst), self.precision)

    def reveal(self, dst=None):
        """Decrypts the tensor as is."""

        # Edge case where share becomes 0 sized (e.g. result of split)
        if self.share.size < 1:
            return jnp.empty(self.share.shape)

        if dst is None:
            array = comm.get().all_reduce(self.share)
        else:
            array = comm.get().reduce(self.share, dst)

        return array

    def copy(self):
        result = Array(src=SENTINEL)
        result.share = self.share.copy()
        result.precision = self.precision
        result.device = self.device
        return result

    def __add__(self, other):
        """Adds two arrays."""
        result = self.copy()
        # Other is Private
        if type(other) is Array:
            share = other.share.copy()
            if self.precision > other.precision:
                scale = 2 ** (self.precision - other.precision)
                share = share * scale
            elif self.precision < other.precision:
                scale = 2 ** (other.precision - self.precision)
                result.share *= scale
                result.precision = other.precision
        # Other is Public
        elif comm.get().get_rank() == 0:
            share = encode(other, result.precision, result.device)
        else:
            share = jnp.zeros(other.shape, dtype=DATATYPE)

        result.share += share
        return result

    def __radd__(self, other):
        """Adds two arrays."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtracts two arrays."""
        return self + (-other)

    def __rsub__(self, other):
        """Subtracts two arrays."""
        return other + (-self)

    def __neg__(self):
        """Negates arrays."""
        result = self.copy()
        result.share *= -1
        return result

    def __mul__(self, other):
        """Multiplies two arrays."""
        if type(other) is Array:
            left = self.copy()
            right = other.copy()

            a, b, c = generate_mul_triple(
                left.share.shape,
                right.share.shape,
                device=self.device,
            )

            left.share -= a.share
            right.share -= b.share

            epsilon = left.reveal()
            delta = right.reveal()

            z = a.share * delta + epsilon * b.share
            if comm.get().get_rank() == 0:
                z += epsilon * delta
            result = Array(src=SENTINEL)
            result.share = c.share + z
            result.precision = self.precision + other.precision
            result.device = self.device
        else:
            result = self.copy()
            other = encode(other, result.precision, result.device)
            result.share = result.share * other
            result.precision *= 2
        return result

    def __rmul__(self, other):
        """Matrix multiplies two arrays."""
        if type(other) is Array:
            result = other * self
        else:
            result = self.copy()
            other = encode(other, result.precision, result.device)
            result.share = other * result.share
            result.precision *= 2
        return result

    def __matmul__(self, other):
        """Matrix multiplies two arrays."""
        if type(other) is Array:
            left = self.copy()
            right = other.copy()

            a, b, c = generate_matmul_triple(
                left.share.shape,
                right.share.shape,
                device=self.device,
            )

            left.share -= a.share
            right.share -= b.share

            epsilon = left.reveal()
            delta = right.reveal()

            z = a.share @ delta + epsilon @ b.share
            if comm.get().get_rank() == 0:
                z += epsilon @ delta
            result = Array(src=SENTINEL)
            result.share = c.share + z
            result.precision = self.precision + other.precision
            result.device = self.device
        else:
            result = self.copy()
            other = encode(other, result.precision, result.device)
            result.share = result.share @ other
            result.precision *= 2
        return result

    def __rmatmul__(self, other):
        """Matrix multiplies two arrays."""
        if type(other) is Array:
            result = other @ self
        else:
            result = self.copy()
            other = encode(other, result.precision, result.device)
            result.share = other @ result.share
            result.precision *= 2
        return result


def generate_mul_triple(shape0, shape1, device=None):
    """Generate multiplicative triples of given shapes"""
    a = random(0, shape0)
    b = random(1, shape1)

    c = a * b

    a = Array(a, precision=0, src=0, device=device)
    b = Array(b, precision=0, src=0, device=device)
    c = Array(c, precision=0, src=0, device=device)

    return a, b, c


def generate_matmul_triple(shape0, shape1, device=None):
    """Generate multiplicative triples of given shapes"""
    a = random(0, shape0)
    b = random(1, shape1)

    c = a @ b

    a = Array(a, precision=0, src=0, device=device)
    b = Array(b, precision=0, src=0, device=device)
    c = Array(c, precision=0, src=0, device=device)

    return a, b, c
