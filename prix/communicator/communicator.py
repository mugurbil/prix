#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import timeit

import numpy as np


class Communicator:
    """
    Abstract class defining the functions that a Communicator should implement.
    """

    BYTES_PER_ELEMENT = 8
    __verbosity = False

    @classmethod
    def is_verbose(cls):
        return cls.__verbosity

    @classmethod
    def set_verbosity(cls, verbosity):
        assert isinstance(verbosity, bool), "Verbosity must be a boolean value"
        cls.__verbosity = verbosity

    @classmethod
    def is_initialized(cls):
        """Returns whether the communicator has been initialized."""
        raise NotImplementedError("is_initialized is not implemented")

    @classmethod
    def get(cls):
        """Returns an instance of the communicator."""
        raise NotImplementedError("get is not implemented")

    @classmethod
    def initialize(cls, **kwargs):
        """Initializes the communicator. Call this function before using it."""
        raise NotImplementedError("initialize is not implemented")

    @classmethod
    def shutdown(cls):
        raise NotImplementedError("shutdown is not implemented")

    def send(self, array, dst):
        """Sends the specified array to the destination dst."""
        raise NotImplementedError("send is not implemented")

    def recv(self, array=None, src=None):
        """Receives an array from an (optional) source src."""
        raise NotImplementedError("recv is not implemented")

    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of arrays to all parties."""
        raise NotImplementedError("scatter is not implemented")

    def reduce(self, array, dst, op="sum", batched=False, async_op=False):
        """Reduces the array data across all parties."""
        raise NotImplementedError("reduce is not implemented")

    def all_reduce(self, array, op="sum", batched=False, async_op=False):
        """Reduces the array data across all parties; all get the final result."""
        raise NotImplementedError("all_reduce is not implemented")

    def gather(self, array, dst, async_op=False):
        """Gathers a list of arrays in a single party."""
        raise NotImplementedError("gather is not implemented")

    def all_gather(self, array, async_op=False):
        """Gathers arrays from all parties in a list."""
        raise NotImplementedError("all_gather is not implemented")

    def broadcast(self, array, src, group=None, async_op=False):
        """Broadcasts the array to all parties."""
        raise NotImplementedError("broadcast is not implemented")

    def barrier(self):
        """Synchronizes all processes."""
        raise NotImplementedError("barrier is not implemented")

    def send_obj(self, obj, dst, group=None):
        """Sends the specified object to the destination `dst`."""
        raise NotImplementedError("send_obj is not implemented")

    def recv_obj(self, src, group=None):
        """Receives an object from a source `src`."""
        raise NotImplementedError("recv_obj is not implemented")

    def broadcast_obj(self, obj, src, group=None):
        """Broadcasts a given object to all parties."""
        raise NotImplementedError("broadcast_obj is not implemented")

    def get_world_size(self):
        """Returns the size of the world."""
        raise NotImplementedError("get_world_size is not implemented")

    def get_evaluators_size(self):
        """Returns the number of evaluator processes."""
        return getattr(self, "evaluator_size", 0)

    def get_evaluator_size(self):
        """Alias for get_evaluators_size."""
        return self.get_evaluators_size()

    def get_ttp_rank(self):
        """Returns the rank of the Trusted Third Party (TTP) server."""
        return self.get_world_size()

    def get_rank(self):
        """Returns the rank of the current process."""
        raise NotImplementedError("get_rank is not implemented")

    def set_name(self, name: str):
        """Sets the party name of the current process."""
        self._name = name

    def get_name(self):
        """Returns the party name of the current process."""
        return getattr(self, "_name", f"rank_{self.get_rank()}")

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.comm_rounds = 0
        self.comm_bytes = 0
        self.comm_time = 0.0
        self.eval_comm_rounds = 0
        self.eval_comm_bytes = 0
        self.eval_comm_time = 0.0

    def print_communication_stats(self):
        """Prints communication statistics."""
        import prix

        prix.log("====Communication Stats====")
        prix.log(f"MPC  Rounds: {self.comm_rounds}")
        prix.log(f"MPC  Bytes : {self.comm_bytes}")
        prix.log(f"MPC  Comm time: {self.comm_time}")
        prix.log(f"Eval Rounds: {self.eval_comm_rounds}")
        prix.log(f"Eval Bytes : {self.eval_comm_bytes}")
        prix.log(f"Eval Comm time: {self.eval_comm_time}")

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.comm_rounds += 1
        self.comm_bytes += int(nelement) * self.BYTES_PER_ELEMENT

    def _log_eval_communication(self, nelement):
        """Updates log of evaluator communication statistics."""
        self.eval_comm_rounds += 1
        self.eval_comm_bytes += int(nelement) * self.BYTES_PER_ELEMENT

    def _log_communication_time(self, comm_time):
        self.comm_time += comm_time

    def _log_eval_communication_time(self, comm_time):
        self.eval_comm_time += comm_time


def _num_elements(x):
    if x is None:
        return 0
    if hasattr(x, "share"):
        return int(x.share.size)
    if hasattr(x, "size"):
        return int(x.size() if callable(x.size) else x.size)
    if isinstance(x, (list, tuple)):
        return sum(_num_elements(item) for item in x)
    return 1


def _logging(func):
    """Decorator that performs logging of communication statistics."""

    @functools.wraps(func)
    def logging_wrapper(self, *args, **kwargs):
        if self.is_verbose():
            if func.__name__ == "barrier":
                self._log_communication(0)
            elif func.__name__ == "scatter":
                n_elem = _num_elements(args[0][0]) if args and args[0] else 0
                self._log_communication(n_elem * max(0, len(args[0]) - 1))
            elif func.__name__ == "broadcast_parallel":
                for arg in args[0]:
                    self._log_eval_communication(_num_elements(arg))
            elif func.__name__ == "send_obj":
                self._log_eval_communication(3)
            elif kwargs.get("batched", False):
                nbytes = sum(_num_elements(x) for x in args[0])
                self._log_communication(nbytes)
            else:
                first_arg = args[0] if args else None
                self._log_communication(_num_elements(first_arg))

            tic = timeit.default_timer()
            result = func(self, *args, **kwargs)
            toc = timeit.default_timer()

            if func.__name__ in ("broadcast_parallel", "send_obj"):
                self._log_eval_communication_time(toc - tic)
            else:
                self._log_communication_time(toc - tic)
            return result

        return func(self, *args, **kwargs)

    return logging_wrapper
