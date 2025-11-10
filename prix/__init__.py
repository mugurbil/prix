#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.0.1"

import logging
import os

import jax
import prix.communicator as comm
from prix.array import Array


def init(rank, world_size, evaluator_size, mailbox, barrier, device=None):
    """
    Initialize Prix. It will initialize communicator, setup party
    name for file save / load, and setup seeds for Random Number Generation.
    By default, the function will initialize a set of RNG generators on CPU.
    If torch.cuda.is_available() returns True, it will initialize an additional
    set of RNG generators on GPU. Users can specify the GPU device the generators are
    initialized with device.

    Args:
        rank (int): party_name for file save and load, default is None
        device (int, str, torch.device): Specify device for RNG generators on
        GPU. Must be a GPU device.
    """
    # jax.config.update("jax_enable_x64", True)

    # Return and raise warning if initialized
    if comm.is_initialized():
        logging.info(f"Comm is initialized for {rank}")
        return

    # Initialize communicator
    # os.environ["GLOO_SOCKET_IFNAME"] = "en0"

    comm._init(
        rank,
        world_size,
        evaluator_size,
        mailbox,
        barrier,
        init_ttp=False,
    )

    # if comm.get().get_evaluators_size() == 0:
    #     for k, v in cfg.config.functions.items():
    #         v = dict(v)
    #         if v['method'] == 'fission':
    #             logging.error(f"Fission needs at least one evaluator for {k}")
    #             exit()

    # Setup seeds for Random Number Generation
    rank = comm.get().get_rank()
    # if comm.get().get_rank() < comm.get().get_world_size():
    # _setup_prng(device)
    # if cfg.mpc.provider == "TTP":
    #     curl.mpc.provider.ttp_provider.TTPClient._init(device)

    # if comm.get().get_evaluators_size() > 0:
    #     prix.evaluator.EvaluatorClient._init()


def init_thread(rank, world_size, evaluator_size=0, device="cpu"):
    jax.config.update("jax_platforms", "cpu")
    comm._init(
        use_threads=True,
        rank=rank,
        world_size=world_size,
        evaluator_size=evaluator_size,
    )


def uninit():
    return comm.uninit()


def array(*args, **kwargs):
    return Array(*args, **kwargs)


__all__ = [
    "init",
    "uninit",
    "array",
]
