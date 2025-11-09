#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp

import prix.communicator as comm
from prix.encoder import encode, decode

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
        array: jnp.array,
        src: int,
        precision: int = 16,
        device=None,
    ):
        """
        Creates the shared array from the input `array` provided by party `src`.

        The other parties can specify an `array` or `size` to determine the size
        of the shared array object to create. In this case, all parties must
        specify the same array size to prevent the party's shares from varying
        in size, which leads to undefined behavior.

        Alternatively, the parties can set `broadcast_size` to `True` to have the
        `src` party broadcast the correct size. The parties who do not know the
        array size beforehand can provide an empty array as input. This is
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
        if self.rank == src:
            assert array is not None, "source must provide a data array"
            if hasattr(array, "src"):
                assert (
                    array.src == src
                ), "source of data array must match source of encryption"

        # If device is unspecified, try and get it from array:
        if device is None and array is not None and hasattr(array, "device"):
            device = array.device

        self.precision = precision

        if self.rank == src:
            # Encode the array
            if array is not None:
                # if jnp.issubdtype(array.dtype, jnp.integer) and precision != 0:
                #     array = array.float()
                array = encode(array, precision=precision, device=device)
                array = jax.device_put(array, device)
            array_sum = array.sum()

            # Secret share the array
            for i in range(comm.get().get_world_size()):
                if i == self.rank:
                    continue
                share_i = jax.random.randint(
                    jax.random.PRNGKey(array_sum + i),
                    array.shape,
                    minval=jax.dtypes.iinfo(jnp.int32).min,
                    maxval=jax.dtypes.iinfo(jnp.int32).max,
                    dtype=jnp.int32,
                )
                comm.get().send(share_i, i)
                array = array - share_i
        else:
            print("RECEIVING")
            array = encode(array, precision=precision, device=device)
            print(f"ARRAY={array}")
            array = comm.get().recv(array, src)
            print(f"COMM ARRAY={array}")

        self._share = array

    @property
    def rank(self):
        return comm.get().get_rank()

    @property
    def share(self):
        return self._share

    def reconstruct(self, dst=None):
        """Decrypts the tensor."""

        # Edge case where share becomes 0 sized (e.g. result of split)
        if self.share.size < 1:
            return jnp.empty(self.share.shape)

        if dst is None:
            array = comm.get().all_reduce(self.share)
        else:
            array = comm.get().reduce(self.share, dst)

        return decode(array, self.precision)
