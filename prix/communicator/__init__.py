#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from .communicator import Communicator
from .distributed_communicator import DistributedCommunicator
from .process_communicator import InProcessCommunicator

_use_distributed = False


def _get_cls():
    return DistributedCommunicator if _use_distributed else InProcessCommunicator


def get():
    cls = _get_cls()
    if not cls.is_initialized():
        raise RuntimeError("Prix not initialized. Please call prix.init() first.")
    return cls.get()


def _init(
    rank=0,
    world_size=1,
    evaluator_size=0,
    mailbox=None,
    barrier=None,
    init_ttp=False,
    distributed=False,
    use_distributed=False,
    coordinator_address=None,
    **kwargs,
):
    global _use_distributed
    distributed = distributed or use_distributed
    _use_distributed = distributed
    cls = _get_cls()

    if cls.is_initialized():
        logging.info("Communicator is already initialized")
        return

    if distributed:
        cls.initialize(
            rank=rank,
            world_size=world_size,
            evaluator_size=evaluator_size,
            coordinator_address=coordinator_address,
            init_ttp=init_ttp,
        )
    else:
        cls.initialize(
            rank=rank,
            world_size=world_size,
            evaluator_size=evaluator_size,
            mailbox=mailbox,
            barrier=barrier,
            init_ttp=init_ttp,
        )


def uninit():
    cls = _get_cls()
    cls.shutdown()


def is_initialized():
    cls = _get_cls()
    return cls.is_initialized()


__all__ = [
    "Communicator",
    "InProcessCommunicator",
    "DistributedCommunicator",
    "get",
    "_init",
    "uninit",
    "is_initialized",
]
