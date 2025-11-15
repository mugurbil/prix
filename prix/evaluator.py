#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import jax
import jax.numpy as jnp

import prix
import prix.communicator as comm
from prix.encoder import PRECISION, DATATYPE
from prix.random import random


class EvaluatorClient:
    __instance = None

    class ___EvaluatorClient:
        """Singleton class"""

        def __init__(self):
            # Initialize connection
            communicator = comm.get()
            self.eval_groups = [[[i, communicator.world_size + j] for j in range(communicator.evaluator_size)] for i in range(communicator.world_size)]
            self.eval_comm_group = [communicator.world_size + i for i in range(communicator.evaluator_size)]
            logging.info(
                f"[Party {communicator.get_rank()}] Evaluator Client initialized"
            )

        def evaluator_request(self, func_name, array, *args, **kwargs):
            communicator = comm.get()
            world_size = communicator.get_world_size()
            evaluators_size = communicator.get_evaluators_size()  # Number of processes
            mpc_party_rank = communicator.get_rank()
            assert (
                communicator.get_rank() < world_size
            ), "Only MPC parties communicate with the EvaluatorServers"

            # Scatter: Divide data into chunks for workers
            chunks = array.chunk(evaluators_size)
            chunks = list(chunks) + [
                prix.array([]) for _ in range(evaluators_size - len(chunks))
            ]
            if mpc_party_rank == 0:
                message = {
                    "function": func_name,
                    "scale": array.encoder.scale,
                }
                for i in range(evaluators_size):
                    evaluator_rank = world_size + i
                    message["array_size"] = chunks[i].size()
                    communicator.send_obj(message, evaluator_rank, self.eval_comm_group)

            # Process each split asynchronously
            communicator.broadcast_parallel(
                chunks, [mpc_party_rank] * len(chunks), self.eval_groups[mpc_party_rank]
            )

            # Initialize local results with the correct split sizes
            results = [
                jnp.empty_like(chunks[i].share, device=array.device)
                for i in range(evaluators_size)
            ]
            communicator.broadcast_parallel(
                results,
                [world_size + i for i in range(evaluators_size)],
                self.eval_groups[mpc_party_rank],
            )

            array.share = jnp.concatenate(results)
            array.encoder._precision_bits = PRECISION
            return array

    @staticmethod
    def _init():
        """Initializes a Evaluator client that sends requests"""
        if EvaluatorClient.__instance is None:
            EvaluatorClient.__instance = EvaluatorClient.___EvaluatorClient()

    @staticmethod
    def uninit():
        """Uninitializes an Evaluator client"""
        del EvaluatorClient.__instance
        EvaluatorClient.__instance = None

    @staticmethod
    def get():
        """Returns the instance of the EvaluatorClient"""
        if EvaluatorClient.__instance is None:
            raise RuntimeError("EvaluatorClient is not initialized")

        return EvaluatorClient.__instance


class EvaluatorServer:
    TERMINATE = -1

    def __init__(self):
        """Initializes an Evaluator server that receives requests"""
        evaluator_rank = comm.get().get_rank()

        # Initialize connection
        logging.info(f"[Evaluator {evaluator_rank}]: Initializing...")
        communicator = comm.get()
        self.eval_groups = [[[i, communicator.world_size + j] for j in range(communicator.evaluator_size)] for i in range(communicator.world_size)]
        self.eval_comm_group = [communicator.world_size + i for i in range(communicator.evaluator_size)]

        # Determine device
        self.device = "cpu"
        logging.info(
            f"[Evaluator {evaluator_rank}] Initialized with device: {self.device}"
        )
        evaluator_rank = communicator.get_rank()
        world_size = communicator.get_world_size()

        # Operations supported by Fission
        fission_operations = {
            "exp": jnp.exp,
            "log": jnp.log,
            "reciprocal": jnp.reciprocal,
            "inv_sqrt": lambda x: 1 / jnp.sqrt(x),
            "sqrt": jnp.sqrt,
            "cos": jnp.cos,
            "sin": jnp.sin,
            "sigmoid": lambda x: 1 / (1 + jnp.exp(-x)),
            "tanh": jnp.tanh,
            "erf": jax.scipy.special.erf,
            "gelu": jax.nn.gelu,
            "silu": jax.nn.silu,
            "softmax": lambda x: jax.nn.softmax(x, dim=-1),
            "log_softmax": lambda x: jax.nn.log_softmax(x, dim=-1),
            "relu": jax.nn.relu,
        }
        try:
            while True:
                # Wait for next request from client
                # Receive the function to evaluate
                message = communicator.recv_obj(0, self.eval_comm_group)
                logging.debug(
                    f"Evaluator [{evaluator_rank}] Message received: %s" % message
                )

                if message == "terminate":
                    logging.info(
                        f"Evaluator Server {evaluator_rank - world_size} shutting down."
                    )
                    exit()
                function = str(message["function"])
                scale = message["scale"]
                array_size = message["array_size"]

                # Receive data from all the MPC nodes
                results = [
                    jnp.empty(array_size, dtype=DATATYPE) for _ in range(world_size)
                ]
                communicator.broadcast_parallel(
                    results,
                    [mpc_node for mpc_node in range(world_size)],
                    [
                        self.eval_groups[mpc_node][evaluator_rank - world_size]
                        for mpc_node in range(world_size)
                    ],
                )

                # Reconstruct
                array = sum(results)
                array = float(array) / scale
                if function == "layernorm":
                    mean = array.mean(-1, keepdims=True)
                    variance = array.var(-1, keepdims=True)
                    inv_var = 1.0 / jax.sqrt(variance + 1e-05)
                    inv_var = inv_var.reshape(mean.shape)
                    # compute z-scores:
                    result = (array - mean) * inv_var
                elif function in fission_operations:
                    result = fission_operations[function](array)
                else:
                    raise ValueError(f"Unsupported function {function}")

                # Secret share the result back to the MPC nodes.
                results = [(result * 2**PRECISION).long()]
                for mpc_node in range(1, world_size):
                    share = random(results[0].sum(), results[0].shape())
                    results[0] -= share
                    results.append(share)

                communicator.broadcast_parallel(
                    results,
                    [evaluator_rank] * len(results),
                    [
                        self.eval_groups[mpc_node][evaluator_rank - world_size]
                        for mpc_node in range(world_size)
                    ],
                )

        except RuntimeError as err:
            logging.info("Encountered Runtime error. Evaluator Server shutting down:")
            logging.info(f"{err}")
