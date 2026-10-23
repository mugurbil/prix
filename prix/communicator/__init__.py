#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from .communicator import Communicator
from .distributed_communicator import DistributedCommunicator
from .process_communicator import InProcessCommunicator
from .tcp_communicator import TCPCommunicator

_TRANSPORTS = {
    "shm": InProcessCommunicator,
    "tcp": TCPCommunicator,
    "jax": DistributedCommunicator,
}
_transport = "shm"


def _get_cls():
    return _TRANSPORTS[_transport]


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
    transport=None,
    peers=None,
    tls=None,
    session=None,
    **kwargs,
):
    """
    Selects and initializes the communicator. `transport` is "shm" (single-host processes,
    the default), "tcp" (direct peer-to-peer TCP, optionally TLS; needs `peers`), or "jax"
    (legacy: messages relayed through the jax.distributed coordinator).
    """
    global _transport
    if transport is None:
        transport = "jax" if (distributed or use_distributed) else "shm"
    if transport not in _TRANSPORTS:
        raise ValueError(f"Unknown transport {transport!r}; choose from {sorted(_TRANSPORTS)}")
    _transport = transport
    cls = _get_cls()

    if cls.is_initialized():
        logging.info("Communicator is already initialized")
        return

    if transport == "tcp":
        cls.initialize(
            rank=rank,
            world_size=world_size,
            evaluator_size=evaluator_size,
            init_ttp=init_ttp,
            peers=peers,
            tls=tls,
            session=session,
        )
    elif transport == "jax":
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
    "TCPCommunicator",
    "get",
    "_init",
    "uninit",
    "is_initialized",
]
