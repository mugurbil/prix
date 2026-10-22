#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

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
from prix.encoder import DATATYPE
from prix.random import PRNGGenerator, generate_random_ring_element, generate_random_ring_element_np


def _np_erf(x):
    if _scipy_erf is not None:
        return _scipy_erf(x)
    return np.asarray(jax.scipy.special.erf(jnp.asarray(x)))


def _np_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _np_log_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


class EvaluatorClient:
    __instance = None

    class _EvaluatorClient:
        """Singleton client for Fission evaluator servers."""

        def __init__(self):
            communicator = comm.get()
            self.eval_groups = communicator.eval_groups
            self.eval_comm_group = communicator.eval_comm_group
            logging.info(
                f"[Party {communicator.get_rank()}] Evaluator Client initialized"
            )

        def evaluator_request(self, func_name, array, *args, **kwargs):
            communicator = comm.get()
            world_size = communicator.get_world_size()
            evaluators_size = communicator.get_evaluators_size()
            mpc_party_rank = communicator.get_rank()
            assert (
                mpc_party_rank < world_size
            ), "Only MPC parties communicate with the EvaluatorServers"

            base_eval_rank = world_size + (1 if communicator.ttp_initialized else 0)

            if evaluators_size == 1:
                evaluator_rank = base_eval_rank
                if mpc_party_rank == 0:
                    message = {
                        "function": func_name,
                        "scale": array.scale,
                        "array_size": array.size(),
                    }
                    communicator.send_obj(message, evaluator_rank, self.eval_comm_group)
                communicator.send(array.share, evaluator_rank)
                recv_share = communicator.recv(array.share, src=evaluator_rank)
                out = array.shallow_copy()
                out.share = recv_share
                out.precision = cfg.encoder.precision_bits
                return out

            # Scatter: Divide data into chunks for evaluator workers
            chunks = list(array.chunk(evaluators_size, dim=0))
            empty_shape = (0,) + array.shape[1:]
            while len(chunks) < evaluators_size:
                empty_arr = array.shallow_copy()
                empty_arr.share = jnp.empty(empty_shape, dtype=DATATYPE)
                chunks.append(empty_arr)

            if mpc_party_rank == 0:
                for i in range(evaluators_size):
                    evaluator_rank = base_eval_rank + i
                    message = {
                        "function": func_name,
                        "scale": array.scale,
                        "array_size": chunks[i].size(),
                    }
                    communicator.send_obj(message, evaluator_rank, self.eval_comm_group)

            # Send each party's chunk share to the corresponding evaluator
            communicator.broadcast_parallel(
                chunks,
                [mpc_party_rank] * len(chunks),
                self.eval_groups[mpc_party_rank],
            )

            # Receive secret shares of the evaluated result from each evaluator
            results = [
                jnp.empty_like(chunks[i].share)
                for i in range(evaluators_size)
            ]
            communicator.broadcast_parallel(
                results,
                [base_eval_rank + i for i in range(evaluators_size)],
                self.eval_groups[mpc_party_rank],
            )

            non_empty = [r for r in results if r.size > 0]
            out = array.clone()
            out.share = jnp.concatenate(non_empty, axis=0) if non_empty else results[0]
            out.precision = cfg.encoder.precision_bits
            return out

    @staticmethod
    def _init():
        """Initializes an Evaluator client that sends requests."""
        if EvaluatorClient.__instance is None:
            EvaluatorClient.__instance = EvaluatorClient._EvaluatorClient()

    @staticmethod
    def uninit():
        """Uninitializes the Evaluator client and shuts down EvaluatorServers."""
        if EvaluatorClient.__instance is not None:
            communicator = comm.get()
            if communicator.get_rank() == 0:
                world_size = communicator.get_world_size()
                evaluators_size = communicator.get_evaluators_size()
                base_eval_rank = world_size + (1 if communicator.ttp_initialized else 0)
                for i in range(evaluators_size):
                    communicator.send_obj(
                        "terminate",
                        base_eval_rank + i,
                        EvaluatorClient.__instance.eval_comm_group,
                    )
            EvaluatorClient.__instance = None

    @staticmethod
    def get():
        """Returns the instance of the EvaluatorClient."""
        if EvaluatorClient.__instance is None:
            raise RuntimeError("EvaluatorClient is not initialized")
        return EvaluatorClient.__instance


class EvaluatorServer:
    TERMINATE = -1

    def __init__(self):
        """Initializes an Evaluator server that receives requests and evaluates nonlinearities."""
        communicator = comm.get()
        evaluator_rank = communicator.get_rank()
        world_size = communicator.get_world_size()
        ttp_offset = 1 if communicator.ttp_initialized else 0
        eval_idx = evaluator_rank - world_size - ttp_offset

        self.generator = PRNGGenerator(seed=12345 + evaluator_rank)
        self.eval_groups = communicator.eval_groups
        self.eval_comm_group = communicator.eval_comm_group
        self.device = "cpu"

        logging.info(
            f"[Evaluator {evaluator_rank}] Initialized with device: {self.device}"
        )

        inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
        fission_operations = {
            "exp": np.exp,
            "log": np.log,
            "reciprocal": np.reciprocal,
            "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
            "sqrt": np.sqrt,
            "cos": np.cos,
            "sin": np.sin,
            "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
            "tanh": np.tanh,
            "erf": _np_erf,
            "gelu": lambda x: 0.5 * x * (1.0 + _np_erf(x * inv_sqrt2)),
            "silu": lambda x: x / (1.0 + np.exp(-x)),
            "softmax": _np_softmax,
            "log_softmax": _np_log_softmax,
            "relu": lambda x: np.maximum(x, 0.0),
        }

        try:
            while True:
                message = communicator.recv_obj(0, self.eval_comm_group)
                logging.debug(
                    f"Evaluator [{evaluator_rank}] Message received: {message}"
                )

                if message == "terminate":
                    logging.info(f"Evaluator Server {eval_idx} shutting down.")
                    return

                function = str(message["function"])
                scale = message["scale"]
                array_size = tuple(message["array_size"])

                # Receive shares directly from mailbox as NumPy arrays (no GPU upload needed)
                results = []
                for mpc_node in range(world_size):
                    _, payload = communicator._get_from_mailbox(src=mpc_node)
                    results.append(np.asarray(payload, dtype=np.int64))

                if results[0].size == 0:
                    out_shares = [
                        np.empty(array_size, dtype=np.int64) for _ in range(world_size)
                    ]
                else:
                    # Reconstruct plaintext in float32
                    raw_sum = results[0] + results[1] if world_size == 2 else np.add.reduce(results)
                    array = raw_sum.astype(np.float32) / np.float32(scale)

                    with np.errstate(all="ignore"):
                        if function == "layernorm":
                            mean = np.mean(array, axis=-1, keepdims=True)
                            variance = np.var(array, axis=-1, keepdims=True)
                            result = (array - mean) / np.sqrt(variance + np.float32(1e-05))
                        elif function in fission_operations:
                            result = fission_operations[function](array)
                        else:
                            raise ValueError(f"Unsupported function {function}")

                        # Secret share result back to the MPC parties
                        encoded = np.round(result * (2 ** cfg.encoder.precision_bits)).astype(
                            np.int64
                        )
                    out_shares = [encoded]
                    for _ in range(1, world_size):
                        share = generate_random_ring_element_np(
                            encoded.shape, generator=self.generator
                        )
                        out_shares[0] = out_shares[0] - share
                        out_shares.append(share)

                for mpc_node in range(world_size):
                    communicator._put_to_mailbox(mpc_node, out_shares[mpc_node])

        except RuntimeError as err:
            logging.info("Encountered Runtime error. Evaluator Server shutting down:")
            logging.info(f"{err}")
