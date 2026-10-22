#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import jax
import jax.numpy as jnp

import prix
import prix.communicator as comm
from prix.config import cfg
from prix.encoder import DATATYPE
from prix.random import PRNGGenerator, generate_random_ring_element


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

        fission_operations = {
            "exp": jnp.exp,
            "log": jnp.log,
            "reciprocal": jnp.reciprocal,
            "inv_sqrt": jax.lax.rsqrt,
            "sqrt": jnp.sqrt,
            "cos": jnp.cos,
            "sin": jnp.sin,
            "sigmoid": jax.nn.sigmoid,
            "tanh": jnp.tanh,
            "erf": jax.scipy.special.erf,
            "gelu": lambda x: jax.nn.gelu(x, approximate=False),
            "silu": jax.nn.silu,
            "softmax": lambda x: jax.nn.softmax(x, axis=-1),
            "log_softmax": lambda x: jax.nn.log_softmax(x, axis=-1),
            "relu": jax.nn.relu,
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

                # Receive shares from all MPC parties
                results = [
                    jnp.empty(array_size, dtype=DATATYPE) for _ in range(world_size)
                ]
                communicator.broadcast_parallel(
                    results,
                    [mpc_node for mpc_node in range(world_size)],
                    [self.eval_groups[mpc_node][eval_idx] for mpc_node in range(world_size)],
                )

                if results[0].size == 0:
                    out_shares = [
                        jnp.empty(array_size, dtype=DATATYPE) for _ in range(world_size)
                    ]
                else:
                    # Reconstruct plaintext in float32
                    raw_sum = sum(results)
                    array = raw_sum.astype(jnp.float32) / float(scale)

                    if function == "layernorm":
                        mean = jnp.mean(array, axis=-1, keepdims=True)
                        variance = jnp.var(array, axis=-1, keepdims=True)
                        inv_var = 1.0 / jnp.sqrt(variance + 1e-05)
                        result = (array - mean) * inv_var
                    elif function in fission_operations:
                        result = fission_operations[function](array)
                    else:
                        raise ValueError(f"Unsupported function {function}")

                    # Secret share result back to the MPC parties
                    encoded = jnp.round(result * (2 ** cfg.encoder.precision_bits)).astype(
                        DATATYPE
                    )
                    out_shares = [encoded]
                    for _ in range(1, world_size):
                        share = generate_random_ring_element(
                            encoded.shape, generator=self.generator
                        )
                        out_shares[0] = out_shares[0] - share
                        out_shares.append(share)

                communicator.broadcast_parallel(
                    out_shares,
                    [evaluator_rank] * len(out_shares),
                    [self.eval_groups[mpc_node][eval_idx] for mpc_node in range(world_size)],
                )

        except RuntimeError as err:
            logging.info("Encountered Runtime error. Evaluator Server shutting down:")
            logging.info(f"{err}")
