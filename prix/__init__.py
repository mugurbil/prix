#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.2.0"

import builtins
import contextlib
import logging
import secrets
from types import SimpleNamespace

logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import prix.communicator as comm
from prix.approximations import LookupTables, bernoulli, randn, where as _where_method
from prix.array import Array, IgnoreEncodings, Ptype, ptype
from prix.config import cfg
from prix.evaluator import EvaluatorClient, EvaluatorServer
from prix.provider import (
    TTPClient,
    TTPServer,
    TrustedFirstParty,
    TrustedThirdParty,
    get_default_provider,
    set_default_provider,
)
from prix.random import PRNGGenerator, generate_kbit_random_tensor

generators = {}


def _setup_prng():
    """
    Generate shared random seeds across parties to generate pseudo-random
    sharings of zero (PRZS) without communication.
    """
    global generators
    communicator = comm.get()
    rank = communicator.get_rank()
    world_size = communicator.get_world_size()

    # All seeds are drawn from OS entropy (never hard-coded): a party's "local"
    # seed is private to it, pairwise PRZS seeds are known only to the two
    # adjacent parties, and the "global" seed is known only to the MPC parties
    # (not to the TTP or Fission evaluators).
    if world_size >= 2:
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        # Each party picks the seed it shares with `next_rank` and sends it there;
        # it receives the seed it shares with `prev_rank`. Sends are non-blocking.
        seed_next = secrets.randbits(64)
        communicator.send_obj(seed_next, next_rank)
        seed_prev = int(communicator.recv_obj(prev_rank))

        # Global seed: XOR of every party's contribution (exchanged directly
        # between parties, not through a single dealer).
        my_global = secrets.randbits(64)
        for dst in range(world_size):
            if dst != rank:
                communicator.send_obj(my_global, dst)
        seed_global = my_global
        for src in range(world_size):
            if src != rank:
                seed_global ^= int(communicator.recv_obj(src))
    else:
        seed_prev = seed_next = secrets.randbits(64)
        seed_global = secrets.randbits(64)

    generators["prev"] = PRNGGenerator(seed=seed_prev)
    generators["next"] = PRNGGenerator(seed=seed_next)
    generators["local"] = PRNGGenerator(seed=secrets.randbits(64))
    generators["global"] = PRNGGenerator(seed=seed_global)


def set_mode(mode, device=None):
    """
    Switch PriX's MPC approximation backend in one line:
      - 'lut' / 'default': Curl Wavelet Look-Up Tables + EGK truncation
      - 'wave': Wave (PoPETs 2025) 2-round FSS Wavelet Look-Up Tables (bior(5,3) + deterministic DPF/DCF truncation + Beaver 8-tuple)
      - 'sigma' / 'fss': SIGMA Function Secret Sharing (DPF/DCF comparisons, bit-packed B2A, static-weight correlations, 0th-order Haar FSS LUTs)
      - 'poly': PUMA / Nimbus piecewise low-degree polynomials (no LUTs)
      - 'shaft': SHAFT 1-round Fourier activations + constant-round ODE Softmax
      - 'crypten': Classic CrypTen polynomial / Newton-Raphson / limit approximations
      - 'fission': Fission hybrid plaintext evaluator
    """
    cfg.set_mode(mode)
    if comm.is_initialized() and comm.get().get_rank() < comm.get().get_world_size():
        LookupTables(device=device)
    return cfg


def init(
    rank=0,
    world_size=1,
    evaluator_size=0,
    mailbox=None,
    barrier=None,
    device=None,
    cfg_file=None,
    mode=None,
    use_distributed=False,
    coordinator_address=None,
    transport=None,
    peers=None,
    tls=None,
    session=None,
):
    """
    Initialize PriX: loads config, initializes communicator, PRNG generators,
    Wavelet LookupTables, TTPClient (if configured), and EvaluatorClient (if configured).

    Multi-host runs use `transport="tcp"` with `peers`, one "host:port" per process in rank
    order (MPC parties, then the TTP if `cfg.mpc.provider == "TTP"`, then evaluators), and
    optionally `tls={"certfile": ..., "keyfile": ..., "cafile": ...}` for mutually
    authenticated, encrypted channels. Processes with rank >= world_size should call
    `prix.serve(...)` instead.
    """
    jax.config.update("jax_enable_x64", True)

    if mode is not None and cfg_file is None:
        cfg_file = mode

    if cfg_file is not None:
        cfg.load_config(cfg_file)

    if comm.is_initialized():
        logging.info(f"Comm is already initialized for rank {rank}")
        return

    init_ttp = cfg.mpc.provider == "TTP"
    comm._init(
        rank=rank,
        world_size=world_size,
        evaluator_size=evaluator_size,
        mailbox=mailbox,
        barrier=barrier,
        init_ttp=init_ttp,
        use_distributed=use_distributed,
        coordinator_address=coordinator_address,
        transport=transport,
        peers=peers,
        tls=tls,
        session=session,
    )

    rank = comm.get().get_rank()
    if rank < comm.get().get_world_size():
        _setup_prng()
        LookupTables(device=device)
        if init_ttp:
            TTPClient._init(device=device)
        if comm.get().get_evaluators_size() > 0:
            EvaluatorClient._init()


def init_thread(rank, world_size, evaluator_size=0, device="cpu"):
    jax.config.update("jax_enable_x64", True)
    comm._init(
        use_threads=True,
        rank=rank,
        world_size=world_size,
        evaluator_size=evaluator_size,
    )
    if rank < world_size:
        _setup_prng()
        LookupTables(device=device)


def uninit():
    if comm.is_initialized():
        if comm.get().get_rank() < comm.get().get_world_size():
            TTPClient.uninit()
            EvaluatorClient.uninit()
        comm.uninit()


def is_initialized():
    return comm.is_initialized()


def serve(rank, world_size, evaluator_size=0, cfg_file=None, mode=None, **comm_kwargs):
    """
    Runs a helper process for multi-host deployments: the TTP dealer (rank == world_size when
    `cfg.mpc.provider == "TTP"`) or a Fission evaluator (any later rank). Takes the same
    transport arguments as `prix.init` (e.g. `transport="tcp", peers=[...], tls={...}`) and
    returns when the MPC parties shut down.
    """
    init(rank, world_size, evaluator_size, cfg_file=cfg_file, mode=mode, **comm_kwargs)
    try:
        if rank < world_size:
            raise ValueError("prix.serve is for TTP/evaluator ranks; MPC parties call prix.init")
        if cfg.mpc.provider == "TTP" and rank == world_size:
            TTPServer()
        else:
            EvaluatorServer()
    finally:
        comm.uninit()


def array(*args, **kwargs):
    return Array(*args, **kwargs)


def cryptensor(*args, **kwargs):
    return Array(*args, **kwargs)


def is_encrypted_array(obj):
    return isinstance(obj, Array)


def is_encrypted_tensor(obj):
    return isinstance(obj, Array)


def cat(tensors, dim=0, axis=None):
    return Array.cat(tensors, dim=dim, axis=axis)


def stack(tensors, dim=0, axis=None):
    return Array.stack(tensors, dim=dim, axis=axis)


def where(condition, input_tensor, other):
    if is_encrypted_tensor(input_tensor):
        return input_tensor.where(condition, other)
    elif is_encrypted_tensor(other):
        return other.where(1 - condition, input_tensor)
    elif is_encrypted_tensor(condition):
        return condition * input_tensor + (1 - condition) * other
    return jnp.where(condition, input_tensor, other)


def rand(*sizes, device=None, ptype=Ptype.arithmetic):
    return Array.rand(*sizes, device=device, ptype=ptype)


def zeros(*sizes, device=None, ptype=Ptype.arithmetic):
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    return Array(jnp.zeros(sizes, dtype=jnp.float32), src=0, device=device, ptype=ptype)


def ones(*sizes, device=None, ptype=Ptype.arithmetic):
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    return Array(jnp.ones(sizes, dtype=jnp.float32), src=0, device=device, ptype=ptype)


def full(size, fill_value, device=None, ptype=Ptype.arithmetic):
    if isinstance(size, int):
        size = (size,)
    return Array(jnp.full(tuple(size), fill_value, dtype=jnp.float32), src=0, device=device, ptype=ptype)


def arange(*args, device=None, ptype=Ptype.arithmetic):
    return Array(jnp.arange(*args, dtype=jnp.float32), src=0, device=device, ptype=ptype)


def eye(n, m=None, device=None, ptype=Ptype.arithmetic):
    return Array(jnp.eye(n, M=m, dtype=jnp.float32), src=0, device=device, ptype=ptype)


@contextlib.contextmanager
def no_grad():
    yield


def trace(tracing=True):
    return get_default_provider().trace(tracing=tracing)


def fill_cache():
    return get_default_provider().fill_cache()


def dump_cache(filename):
    return get_default_provider().dump_cache(filename)


def save_cache():
    return get_default_provider().save_cache()


def load_cache():
    return get_default_provider().load_cache()


def print(*args, in_order=False, dst=0, **kwargs):
    """Prints from rank `dst` (default 0) or across all parties in rank order."""
    if not comm.is_initialized():
        builtins.print(*args, **kwargs)
        return
    if in_order:
        for i in range(comm.get().get_world_size()):
            if comm.get().get_rank() == i:
                builtins.print(*args, **kwargs)
            comm.get().barrier()
    elif comm.get().get_rank() == dst:
        builtins.print(*args, **kwargs)


def log(*args, in_order=False, dst=0, level=logging.INFO):
    if not comm.is_initialized():
        logging.log(level, *args)
        return
    if in_order:
        for i in range(comm.get().get_world_size()):
            if comm.get().get_rank() == i:
                logging.log(level, *args)
            comm.get().barrier()
    elif comm.get().get_rank() == dst:
        logging.log(level, *args)


from prix.launcher import MultiProcessLauncher, run, run_multiprocess  # noqa: E402
import prix.nn as nn  # noqa: E402

mpc = SimpleNamespace(
    get_default_provider=get_default_provider,
    set_default_provider=set_default_provider,
    TrustedFirstParty=TrustedFirstParty,
    TrustedThirdParty=TrustedThirdParty,
    TTPClient=TTPClient,
    TTPServer=TTPServer,
    ptype=ptype,
    binary=Ptype.binary,
    arithmetic=Ptype.arithmetic,
    run_multiprocess=run_multiprocess,
)

__all__ = [
    "Array",
    "IgnoreEncodings",
    "LookupTables",
    "MultiProcessLauncher",
    "Ptype",
    "arange",
    "array",
    "bernoulli",
    "cat",
    "cfg",
    "comm",
    "cryptensor",
    "dump_cache",
    "eye",
    "fill_cache",
    "full",
    "generators",
    "init",
    "init_thread",
    "is_encrypted_array",
    "is_encrypted_tensor",
    "is_initialized",
    "load_cache",
    "log",
    "mpc",
    "nn",
    "no_grad",
    "ones",
    "print",
    "ptype",
    "rand",
    "randn",
    "run",
    "run_multiprocess",
    "save_cache",
    "set_mode",
    "stack",
    "trace",
    "uninit",
    "where",
    "zeros",
]
