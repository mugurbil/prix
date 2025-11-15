#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from .communicator import Communicator

# from .distributed_communicator import DistributedCommunicator
from .process_communicator import InProcessCommunicator


def get():
    # cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    cls = InProcessCommunicator

    if not cls.is_initialized():
        raise RuntimeError("Prix not initialized. Please call prix.init() first.")

    return cls.get()


def _init(rank, world_size, evaluator_size, mailbox, barrier, init_ttp=False):
    # cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    cls = InProcessCommunicator

    if cls.is_initialized():
        logging.info(f"Communicator is initialized")
        return

    cls.initialize(
        rank, world_size, evaluator_size, mailbox, barrier, init_ttp=init_ttp
    )


def uninit():
    # cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    cls = InProcessCommunicator
    cls.shutdown()


def is_initialized():
    # cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    cls = InProcessCommunicator
    return cls.is_initialized()


# expose classes and functions in package:
# __all__ = ["Communicator", "DistributedCommunicator", "get", "uninit", "is_initialized"]
__all__ = ["Communicator", "get", "uninit", "is_initialized"]
