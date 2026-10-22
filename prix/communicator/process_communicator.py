#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import threading
from operator import itemgetter

import jax
import jax.numpy as jnp
import numpy as np

from .communicator import Communicator, _logging


class InProcessCommunicator(Communicator):
    """
    In-process / multiprocess communicator using per-rank Queue mailboxes and a Barrier.
    """

    BYTES_PER_ELEMENT = 8
    tls = threading.local()
    mailbox = None
    barrier_obj = None
    lock = threading.Lock()

    @classmethod
    def initialize(
        cls,
        rank,
        world_size,
        evaluator_size=0,
        mailbox=None,
        barrier=None,
        init_ttp=False,
    ):
        if mailbox is not None:
            InProcessCommunicator.mailbox = mailbox
        if barrier is not None:
            InProcessCommunicator.barrier_obj = barrier
        cls.tls.instance = cls(
            rank, world_size, evaluator_size, init_ttp=init_ttp
        )

    def __init__(self, rank, world_size, evaluator_size=0, init_ttp=False):
        self.rank = rank
        self.world_size = world_size
        self.evaluator_size = evaluator_size
        self.ttp_initialized = init_ttp
        self.ttp_offset = 1 if init_ttp else 0
        self._pending = []
        self.reset_communication_stats()
        self._name = f"rank_{rank}"

        # Setup evaluator & TTP groups
        eval_base = self.world_size + self.ttp_offset
        self.eval_groups = [
            [[i, eval_base + j] for j in range(self.evaluator_size)]
            for i in range(self.world_size)
        ]
        self.eval_comm_group = [
            eval_base + i for i in range(self.evaluator_size)
        ]
        self.ttp_group = list(range(self.world_size + 1)) if init_ttp else []
        self.ttp_comm_group = [0, self.world_size] if init_ttp else []

    @classmethod
    def get(cls):
        if not hasattr(cls.tls, "instance"):
            return None
        return cls.tls.instance

    @classmethod
    def is_initialized(cls):
        return hasattr(cls.tls, "instance") and cls.tls.instance is not None

    @classmethod
    def shutdown(cls):
        cls.tls = threading.local()

    def _put_to_mailbox(self, dst, payload):
        self.mailbox[dst].put((self.rank, payload))

    def _get_from_mailbox(self, src=None):
        if src is None:
            if self._pending:
                return self._pending.pop(0)
            return self.mailbox[self.rank].get()

        for idx, (sender_rank, payload) in enumerate(self._pending):
            if sender_rank == src:
                self._pending.pop(idx)
                return sender_rank, payload

        while True:
            sender_rank, payload = self.mailbox[self.rank].get()
            if sender_rank == src:
                return sender_rank, payload
            self._pending.append((sender_rank, payload))

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _default_device():
        return jax.devices()[0]

    @classmethod
    def _get_device(cls, array):
        if hasattr(array, "share"):
            array = array.share
        if isinstance(array, jnp.ndarray):
            devs = array.devices()
            if len(devs) == 1:
                dev = next(iter(devs))
                if dev != cls._default_device():
                    return dev
        return None

    @staticmethod
    def _to_transport(array):
        if hasattr(array, "share"):
            array = array.share
        if isinstance(array, (np.ndarray, jnp.ndarray)):
            return np.asarray(array)
        return array

    @staticmethod
    def _to_transport_batch(arrays):
        raw = [a.share if hasattr(a, "share") else a for a in arrays]
        return [np.asarray(x) for x in jax.device_get(raw)]

    @staticmethod
    def _from_transport(payload, device=None):
        if isinstance(payload, np.ndarray):
            if device is not None:
                return jax.device_put(payload, device)
            return jnp.asarray(payload)
        return payload

    @_logging
    def send_obj(self, obj, dst, group=None):
        """Sends the specified object to the destination dst."""
        self._put_to_mailbox(dst, obj)

    def recv_obj(self, src=None, group=None):
        """Receives an object from an (optional) source src."""
        _, result = self._get_from_mailbox(src=src)
        return result

    @_logging
    def broadcast_obj(self, obj, src, group=None):
        """Broadcasts a Python object from `src` to all parties in `group`."""
        targets = range(self.world_size) if group is None else group
        if self.rank == src:
            for dst in targets:
                if dst != self.rank:
                    self._put_to_mailbox(dst, obj)
            return obj
        else:
            _, result = self._get_from_mailbox(src=src)
            return result

    @_logging
    def send(self, array, dst):
        """Sends the specified array to the destination dst."""
        self._put_to_mailbox(dst, self._to_transport(array))

    @_logging
    def recv(self, array=None, src=None):
        """Receives an array from an (optional) source src."""
        _, result = self._get_from_mailbox(src=src)
        return self._from_transport(result)

    def isend(self, array, dst):
        """Asynchronously sends the specified array to the destination dst."""
        self.send(array, dst)

        class Result:
            def is_completed(self):
                return True

            def wait(self):
                return None

        return Result()

    def irecv(self, array=None, src=None):
        """Asynchronously receives an array from an (optional) source src."""
        comm_self = self

        class Result:
            def __init__(self):
                self.completed = False
                self.value = None

            def is_completed(self):
                return self.completed

            def wait(self):
                if not self.completed:
                    self.value = comm_self.recv(array, src=src)
                    self.completed = True
                return self.value

        return Result()

    @_logging
    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of arrays to all parties."""
        if async_op:
            raise NotImplementedError()

        if src == self.rank:
            for i in range(self.world_size):
                if i != self.rank:
                    self._put_to_mailbox(i, self._to_transport(scatter_list[i]))
            return self._from_transport(self._to_transport(scatter_list[self.rank]))
        else:
            _, payload = self._get_from_mailbox(src=src)
            return self._from_transport(payload)

    def _reduce_arrays(self, arrays, op="sum", device=None):
        if len(arrays) == 2:
            if op == "sum":
                res = arrays[0] + arrays[1]
            elif op in ("xor", "bxor"):
                res = arrays[0] ^ arrays[1]
            else:
                raise NotImplementedError(f"Unsupported reduction op: {op}")
            return self._from_transport(res, device=device)
        if op == "sum":
            res = np.add.reduce(arrays)
        elif op in ("xor", "bxor"):
            res = np.bitwise_xor.reduce(arrays)
        else:
            raise NotImplementedError(f"Unsupported reduction op: {op}")
        return self._from_transport(res, device=device)

    @_logging
    def reduce(self, array, dst, op="sum", batched=False, async_op=False):
        """Reduces the array data across all parties to `dst`."""
        if async_op:
            raise NotImplementedError()

        if batched:
            payload = self._to_transport_batch(array)
        else:
            payload = self._to_transport(array)

        if self.rank != dst:
            self._put_to_mailbox(dst, payload)
            return None
        else:
            gathered = {self.rank: payload}
            for other in range(self.world_size):
                if other == self.rank:
                    continue
                _, other_payload = self._get_from_mailbox(src=other)
                gathered[other] = other_payload

            ordered = [gathered[r] for r in range(self.world_size)]
            if batched:
                num_items = len(array)
                dev = self._get_device(array[0]) if num_items > 0 else None
                return [
                    self._reduce_arrays(
                        [ordered[r][k] for r in range(self.world_size)],
                        op=op,
                        device=dev,
                    )
                    for k in range(num_items)
                ]
            else:
                dev = self._get_device(array)
                return self._reduce_arrays(
                    ordered,
                    op=op,
                    device=dev,
                )

    @_logging
    def all_reduce(self, array, op="sum", batched=False, async_op=False):
        """Reduces the array data across all parties; all get the final result."""
        if async_op:
            raise NotImplementedError()

        if batched:
            payload = self._to_transport_batch(array)
        else:
            payload = self._to_transport(array)

        for i in range(self.world_size):
            if i != self.rank:
                self._put_to_mailbox(i, payload)

        gathered = {self.rank: payload}
        for other in range(self.world_size):
            if other == self.rank:
                continue
            _, other_payload = self._get_from_mailbox(src=other)
            gathered[other] = other_payload

        ordered = [gathered[r] for r in range(self.world_size)]
        if batched:
            num_items = len(array)
            dev = self._get_device(array[0]) if num_items > 0 else None
            return [
                self._reduce_arrays(
                    [ordered[r][k] for r in range(self.world_size)],
                    op=op,
                    device=dev,
                )
                for k in range(num_items)
            ]
        else:
            dev = self._get_device(array)
            return self._reduce_arrays(
                ordered,
                op=op,
                device=dev,
            )

    @_logging
    def gather(self, array, dst, async_op=False):
        """Gathers a list of arrays in a single party."""
        if async_op:
            raise NotImplementedError()

        payload = self._to_transport(array)
        if self.rank != dst:
            self._put_to_mailbox(dst, payload)
            return None
        else:
            gathered = {self.rank: self._from_transport(payload)}
            for other in range(self.world_size):
                if other == self.rank:
                    continue
                _, other_payload = self._get_from_mailbox(src=other)
                gathered[other] = self._from_transport(other_payload)
            return [gathered[r] for r in range(self.world_size)]

    @_logging
    def all_gather(self, array, async_op=False):
        """Gathers arrays from all parties in a list."""
        if async_op:
            raise NotImplementedError()

        payload = self._to_transport(array)
        for i in range(self.world_size):
            if i != self.rank:
                self._put_to_mailbox(i, payload)

        gathered = {self.rank: self._from_transport(payload)}
        for other in range(self.world_size):
            if other == self.rank:
                continue
            _, other_payload = self._get_from_mailbox(src=other)
            gathered[other] = self._from_transport(other_payload)

        return [gathered[r] for r in range(self.world_size)]

    @_logging
    def broadcast(self, array, src, group=None, async_op=False):
        """Broadcasts the array to all parties in `group`."""
        if async_op:
            raise NotImplementedError()

        targets = range(self.world_size) if group is None else group
        if self.rank == src:
            payload = self._to_transport(array)
            for i in targets:
                if i != self.rank:
                    self._put_to_mailbox(i, payload)
            return self._from_transport(payload)
        else:
            _, payload = self._get_from_mailbox(src=src)
            return self._from_transport(payload)

    @_logging
    def broadcast_parallel(self, inputs_list, src_list, groups_list):
        """Broadcasts a list of arrays across corresponding (src, group) pairs."""
        assert (
            len(inputs_list) == len(groups_list) == len(src_list)
        ), f"inputs_list {len(inputs_list)}, src_list {len(src_list)} and groups_list {len(groups_list)} must have same length"

        for idx, (item, src, group) in enumerate(zip(inputs_list, src_list, groups_list)):
            if self.rank == src:
                payload = self._to_transport(item)
                for dst in group:
                    if dst != self.rank:
                        self._put_to_mailbox(dst, payload)
            elif self.rank in group:
                _, payload = self._get_from_mailbox(src=src)
                recv_arr = self._from_transport(payload, device=self._get_device(item))
                if hasattr(item, "share"):
                    item.share = recv_arr
                else:
                    inputs_list[idx] = recv_arr
        return inputs_list

    @_logging
    def barrier(self):
        if self.barrier_obj is not None and self.rank < self.world_size:
            self.barrier_obj.wait()

    def get_world_size(self):
        """Returns the size of the world."""
        return self.world_size

    def get_evaluators_size(self):
        """Returns the number of evaluators."""
        return self.evaluator_size

    def get_rank(self):
        """Returns the rank of the current process."""
        return self.rank

    def get_name(self):
        """Returns the party name of the current rank."""
        return self._name
