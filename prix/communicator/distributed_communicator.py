#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import pickle

import jax
import jax.distributed as jdist
import jax.numpy as jnp
import numpy as np
from jax._src.distributed import global_state

from .communicator import Communicator, _logging


class DistributedCommunicator(Communicator):
    """
    Distributed communicator backed by `jax.distributed` and its coordinator KV store.
    """

    BYTES_PER_ELEMENT = 8
    _instance = None
    _TIMEOUT_MS = 600_000

    @classmethod
    def initialize(
        cls,
        rank=None,
        world_size=None,
        evaluator_size=0,
        coordinator_address=None,
        init_ttp=False,
    ):
        if cls._instance is not None:
            return

        if rank is None:
            rank = int(os.environ.get("JAX_PROCESS_INDEX", os.environ.get("RANK", 0)))
        if world_size is None:
            world_size = int(
                os.environ.get("JAX_PROCESS_COUNT", os.environ.get("WORLD_SIZE", 1))
            )
        if coordinator_address is None:
            coordinator_address = os.environ.get(
                "JAX_COORDINATOR_ADDRESS", os.environ.get("COORDINATOR_ADDRESS", "127.0.0.1:12345")
            )

        total_processes = world_size + (1 if init_ttp else 0) + evaluator_size
        if global_state.client is None and total_processes > 1:
            jdist.initialize(
                coordinator_address=coordinator_address,
                num_processes=total_processes,
                process_id=rank,
            )

        cls._instance = cls(
            rank=rank,
            world_size=world_size,
            evaluator_size=evaluator_size,
            init_ttp=init_ttp,
        )

    def __init__(self, rank, world_size, evaluator_size=0, init_ttp=False):
        self.rank = rank
        self.world_size = world_size
        self.evaluator_size = evaluator_size
        self.ttp_initialized = init_ttp
        self.ttp_offset = 1 if init_ttp else 0
        self._seq_counters = {}
        self._barrier_counter = 0
        self.reset_communication_stats()
        self._name = f"rank_{rank}"

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
        return cls._instance

    @classmethod
    def is_initialized(cls):
        return cls._instance is not None

    @classmethod
    def shutdown(cls):
        if global_state.client is not None:
            try:
                jdist.shutdown()
            except Exception:
                pass
        cls._instance = None

    def _next_seq(self, src, dst):
        key = (src, dst)
        seq = self._seq_counters.get(key, 0)
        self._seq_counters[key] = seq + 1
        return seq

    def _kv_put(self, src, dst, payload):
        seq = self._next_seq(src, dst)
        key = f"prix_msg_{src}_{dst}_{seq}"
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        global_state.client.key_value_set(key, data)

    def _kv_get(self, src, dst):
        seq = self._next_seq(src, dst)
        key = f"prix_msg_{src}_{dst}_{seq}"
        data = global_state.client.blocking_key_value_get(key, self._TIMEOUT_MS)
        return pickle.loads(data)

    @staticmethod
    def _to_transport(array):
        if hasattr(array, "share"):
            array = array.share
        if isinstance(array, (np.ndarray, jnp.ndarray)):
            return np.asarray(array)
        return array

    @staticmethod
    def _from_transport(payload):
        if isinstance(payload, np.ndarray):
            return jnp.asarray(payload)
        return payload

    @_logging
    def send_obj(self, obj, dst, group=None):
        self._kv_put(self.rank, dst, obj)

    def recv_obj(self, src, group=None):
        if src is None:
            raise ValueError("DistributedCommunicator.recv_obj requires an explicit `src` rank.")
        return self._kv_get(src, self.rank)

    @_logging
    def broadcast_obj(self, obj, src, group=None):
        targets = range(self.world_size) if group is None else group
        if self.rank == src:
            for dst in targets:
                if dst != self.rank:
                    self._kv_put(self.rank, dst, obj)
            return obj
        else:
            return self._kv_get(src, self.rank)

    @_logging
    def send(self, array, dst):
        self._kv_put(self.rank, dst, self._to_transport(array))

    @_logging
    def recv(self, array=None, src=None):
        if src is None:
            raise ValueError("DistributedCommunicator.recv requires an explicit `src` rank.")
        return self._from_transport(self._kv_get(src, self.rank))

    def isend(self, array, dst):
        self.send(array, dst)

        class Result:
            def is_completed(self):
                return True

            def wait(self):
                return None

        return Result()

    def irecv(self, array=None, src=None):
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
        if self.rank == src:
            for i in range(self.world_size):
                if i != self.rank:
                    self._kv_put(self.rank, i, self._to_transport(scatter_list[i]))
            return self._from_transport(self._to_transport(scatter_list[self.rank]))
        else:
            return self._from_transport(self._kv_get(src, self.rank))

    def _reduce_arrays(self, arrays, op="sum"):
        stacked = jnp.stack(arrays, axis=0)
        if op == "sum":
            return jnp.sum(stacked, axis=0)
        elif op in ("xor", "bxor"):
            return functools.reduce(jnp.bitwise_xor, arrays)
        else:
            raise NotImplementedError(f"Unsupported reduction op: {op}")

    @_logging
    def reduce(self, array, dst, op="sum", batched=False, async_op=False):
        if batched:
            payload = [self._to_transport(a) for a in array]
        else:
            payload = self._to_transport(array)

        if self.rank != dst:
            self._kv_put(self.rank, dst, payload)
            return None
        else:
            ordered = []
            for r in range(self.world_size):
                if r == self.rank:
                    ordered.append(payload)
                else:
                    ordered.append(self._kv_get(r, self.rank))

            if batched:
                num_items = len(array)
                return [
                    self._reduce_arrays(
                        [self._from_transport(ordered[r][k]) for r in range(self.world_size)],
                        op=op,
                    )
                    for k in range(num_items)
                ]
            else:
                return self._reduce_arrays(
                    [self._from_transport(ordered[r]) for r in range(self.world_size)],
                    op=op,
                )

    @_logging
    def all_reduce(self, array, op="sum", batched=False, async_op=False):
        if batched:
            payload = [self._to_transport(a) for a in array]
        else:
            payload = self._to_transport(array)

        for dst in range(self.world_size):
            if dst != self.rank:
                self._kv_put(self.rank, dst, payload)

        ordered = []
        for r in range(self.world_size):
            if r == self.rank:
                ordered.append(payload)
            else:
                ordered.append(self._kv_get(r, self.rank))

        if batched:
            num_items = len(array)
            return [
                self._reduce_arrays(
                    [self._from_transport(ordered[r][k]) for r in range(self.world_size)],
                    op=op,
                )
                for k in range(num_items)
            ]
        else:
            return self._reduce_arrays(
                [self._from_transport(ordered[r]) for r in range(self.world_size)],
                op=op,
            )

    @_logging
    def gather(self, array, dst, async_op=False):
        payload = self._to_transport(array)
        if self.rank != dst:
            self._kv_put(self.rank, dst, payload)
            return None
        else:
            result = []
            for r in range(self.world_size):
                if r == self.rank:
                    result.append(self._from_transport(payload))
                else:
                    result.append(self._from_transport(self._kv_get(r, self.rank)))
            return result

    @_logging
    def all_gather(self, array, async_op=False):
        payload = self._to_transport(array)
        for dst in range(self.world_size):
            if dst != self.rank:
                self._kv_put(self.rank, dst, payload)

        result = []
        for r in range(self.world_size):
            if r == self.rank:
                result.append(self._from_transport(payload))
            else:
                result.append(self._from_transport(self._kv_get(r, self.rank)))
        return result

    @_logging
    def broadcast(self, array, src, group=None, async_op=False):
        targets = range(self.world_size) if group is None else group
        if self.rank == src:
            payload = self._to_transport(array)
            for dst in targets:
                if dst != self.rank:
                    self._kv_put(self.rank, dst, payload)
            return self._from_transport(payload)
        else:
            return self._from_transport(self._kv_get(src, self.rank))

    @_logging
    def broadcast_parallel(self, inputs_list, src_list, groups_list):
        assert len(inputs_list) == len(groups_list) == len(src_list)
        for idx, (item, src, group) in enumerate(zip(inputs_list, src_list, groups_list)):
            if self.rank == src:
                payload = self._to_transport(item)
                for dst in group:
                    if dst != self.rank:
                        self._kv_put(self.rank, dst, payload)
            elif self.rank in group:
                recv_arr = self._from_transport(self._kv_get(src, self.rank))
                if hasattr(item, "share"):
                    item.share = recv_arr
                else:
                    inputs_list[idx] = recv_arr
        return inputs_list

    @_logging
    def barrier(self):
        if self.world_size <= 1:
            return
        self._barrier_counter += 1
        tag = self._barrier_counter
        for dst in range(self.world_size):
            if dst != self.rank:
                self._kv_put(self.rank, dst, ("barrier", tag))
        for src in range(self.world_size):
            if src != self.rank:
                self._kv_get(src, self.rank)

    def get_world_size(self):
        return self.world_size

    def get_evaluators_size(self):
        return self.evaluator_size

    def get_rank(self):
        return self.rank

    def get_name(self):
        return self._name
