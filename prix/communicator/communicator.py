#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import timeit

import jax


class Communicator:
    """
    Abstract class defining the functions that a Communicator should implement.
    """

    # Determines whether communicators log communication stats
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
        """Returns whether the communicator has been initialized"""
        raise NotImplementedError("is_initialized is not implemented")

    @classmethod
    def get(cls):
        """Returns an instance of the communicator"""
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

    def recv(self, array, src=None):
        """Receives an array from an (optional) source src."""
        raise NotImplementedError("recv is not implemented")

    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of arrays to all parties."""
        raise NotImplementedError("scatter is not implemented")

    def reduce(self, array, dst, async_op=False):
        """Reduces the array data across all parties."""
        raise NotImplementedError("reduce is not implemented")

    def all_reduce(self, array, async_op=False):
        """Reduces the array data across all parties; all get the final result."""
        raise NotImplementedError("all_reduce is not implemented")

    def gather(self, array, dst, async_op=False):
        """Gathers a list of arrays in a single party."""
        raise NotImplementedError("gather is not implemented")

    def all_gather(self, array, async_op=False):
        """Gathers arrays from all parties in a list."""
        raise NotImplementedError("all_gather is not implemented")

    def broadcast(self, array, src, async_op=False):
        """Broadcasts the array to all parties."""
        raise NotImplementedError("broadcast is not implemented")

    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this
        function.
        """
        raise NotImplementedError("barrier is not implemented")

    def send_obj(self, obj, group):
        """Sends the specified object to the destination `dst`."""
        raise NotImplementedError("send_obj is not implemented")

    def recv_obj(self, src, group):
        """Receives an array from a source src."""
        raise NotImplementedError("recv_obj is not implemented")

    def broadcast_obj(self, obj, src):
        """Broadcasts a given object to all parties."""
        raise NotImplementedError("broadcast_obj is not implemented")

    def get_world_size(self):
        """Returns the size of the world."""
        raise NotImplementedError("get_world_size is not implemented")

    def get_evaluator_size(self):
        """Returns the size of the eval."""
        raise NotImplementedError("get_evaluator_size is not implemented")

    def get_rank(self):
        """Returns the rank of the current process."""
        raise NotImplementedError("get_rank is not implemented")

    def set_name(self, name: str):
        """Sets the party name of the current process."""
        raise NotImplementedError("set_name is not implemented")

    def get_name(self):
        """Returns the party name of the current process."""
        raise NotImplementedError("get_name is not implemented")

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.comm_rounds = 0
        self.comm_bytes = 0
        self.comm_time = 0
        self.eval_comm_rounds = 0
        self.eval_comm_bytes = 0
        self.eval_comm_time = 0

    def print_communication_stats(self):
        """Prints communication statistics."""
        import prix

        prix.log("====Communication Stats====")
        prix.log("MPC  Rounds: {}".format(self.comm_rounds))
        prix.log("MPC  Bytes : {}".format(self.comm_bytes))
        prix.log("MPC  Comm time: {}".format(self.comm_time))
        prix.log("Eval Rounds: {}".format(self.eval_comm_rounds))
        prix.log("Eval Bytes : {}".format(self.eval_comm_bytes))
        prix.log("Eval Comm time: {}".format(self.eval_comm_time))

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.comm_rounds += 1
        self.comm_bytes += nelement * self.BYTES_PER_ELEMENT

    def _log_eval_communication(self, nelement):
        """Updates log of communication statistics."""
        self.eval_comm_rounds += 1
        self.eval_comm_bytes += nelement * self.BYTES_PER_ELEMENT

    def _log_communication_time(self, comm_time):
        self.comm_time += comm_time

    def _log_eval_communication_time(self, comm_time):
        self.eval_comm_time += comm_time

    def get_generator(self, idx, device=None):
        """
        Get the corresponding RNG generator, as specified by its index and device

        Args:
            idx: The index of the generator, can be either 0 or 1
            device: The device that the generator lives in.
        """

        if device is None:
            device = jax.device("cpu")
        else:
            device = jax.device(device)

        if idx not in {0, 1}:
            raise RuntimeError(f"Generator idx {idx} out of bounds.")

        generator_name = f"g{idx}_cuda" if device.type == "cuda" else f"g{idx}"
        generator = getattr(self, generator_name, None)

        if generator is None:
            raise ValueError(
                f"Generator {generator_name} is not initialized, call prix.init() first"
            )

        return generator


def _logging(func):
    """Decorator that performs logging of communication statistics."""

    def logging_wrapper(self, *args, **kwargs):

        # only log if needed:
        if self.is_verbose():
            if func.__name__ == "barrier":
                self._log_communication(0, 1)
            elif func.__name__ == "scatter":  # N - 1 arrays communicated
                self._log_communication(args[0][0].nelement() * (len(args[0]) - 1))
            elif func.__name__ == "broadcast_parallel":
                for arg in args[0]:
                    self._log_eval_communication(arg.nelement())
            elif func.__name__ == "send_obj":
                self._log_eval_communication(3)
            elif "batched" in kwargs and kwargs["batched"]:
                nbytes = sum(x.nelement() for x in args[0])
                self._log_communication(nbytes)
            else:  # one array communicated
                self._log_communication(args[0].nelement())

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
