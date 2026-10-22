#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import ctypes
import functools
import logging
import multiprocessing
import multiprocessing.connection
import os
import pickle
import threading
import time
import uuid

import numpy as np

import prix
from prix.communicator.process_communicator import InProcessCommunicator
from prix.config import cfg
from prix.evaluator import EvaluatorServer
from prix.provider import TTPServer


def _to_host_return_value(val):
    """Converts JAX / PriX arrays inside a return value to host NumPy arrays for IPC."""
    if val is None:
        return None
    if hasattr(val, "__jax_array__") or type(val).__module__.startswith("jax"):
        return np.asarray(val)
    if isinstance(val, (tuple, list)):
        converted = [_to_host_return_value(x) for x in val]
        return tuple(converted) if isinstance(val, tuple) else converted
    if isinstance(val, dict):
        return {k: _to_host_return_value(v) for k, v in val.items()}
    return val


_START_METHODS = ("auto", "fork", "spawn", "forkserver")
_SPAWN_NOTICE_SHOWN = False


def _jax_backend_initialized():
    """True if this process has already initialized a JAX/XLA backend (or we can't tell)."""
    try:
        from jax._src import xla_bridge

        return bool(xla_bridge.backends_are_initialized())
    except Exception:
        return True


def _resolve_start_method(start_method):
    """
    Picks the multiprocessing start method for MPC parties.

    'auto' (default) uses 'fork' (fast; works with functions defined in notebooks and
    interactive sessions) unless this process has already initialized JAX. Forking a
    process whose XLA runtime threads are live can deadlock the children, so in that
    case 'auto' falls back to 'spawn'. Override with the `start_method` argument or the
    PRIX_START_METHOD environment variable.
    """
    method = start_method or os.environ.get("PRIX_START_METHOD", "auto")
    if method not in _START_METHODS:
        raise ValueError(f"start_method must be one of {_START_METHODS}, got {method!r}")
    if method == "auto":
        if "fork" in multiprocessing.get_all_start_methods() and not _jax_backend_initialized():
            return "fork"
        global _SPAWN_NOTICE_SHOWN
        if not _SPAWN_NOTICE_SHOWN:
            _SPAWN_NOTICE_SHOWN = True
            logging.info(
                "PriX: JAX is already initialized in this process; starting MPC parties with "
                "'spawn' to avoid fork-after-XLA deadlocks (adds ~2s of startup per run)."
            )
        return "spawn"
    return method


def _config_snapshot():
    """Plain-dict copy of the active PriX config, so children see the parent's settings."""

    def to_plain(node):
        if isinstance(node, dict):
            return {k: to_plain(v) for k, v in node.items()}
        return copy.deepcopy(node)

    return to_plain(cfg.config)


def _dumps(obj):
    """Serializes functions/args for non-fork children; cloudpickle handles notebook-defined code."""
    try:
        import cloudpickle

        return cloudpickle.dumps(obj)
    except ImportError:
        return pickle.dumps(obj)


class MultiProcessLauncher:
    """Launches MPC parties, optional TTP server, and optional Fission Evaluator servers."""

    def __init__(
        self,
        world_size,
        evaluator_size=0,
        run_process_fn=None,
        fn_args=None,
        cfg_file=None,
        mode=None,
        device=None,
        start_method=None,
    ):
        # Support both (world_size, evaluator_size, run_process_fn, fn_args)
        # and (world_size, run_process_fn, fn_args, cfg_file) signatures
        if callable(evaluator_size):
            if cfg_file is None:
                cfg_file = fn_args
            fn_args = run_process_fn
            run_process_fn = evaluator_size
            evaluator_size = getattr(fn_args, "evaluator_size", 0) if fn_args is not None else 0

        if mode is not None and cfg_file is None:
            cfg_file = mode

        if cfg_file is not None:
            cfg.load_config(cfg_file)

        if evaluator_size == 0 and getattr(cfg.encoder.trunc_method, "prod", None) == "fission":
            evaluator_size = 1

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        self.start_method = _resolve_start_method(start_method)
        ctx = multiprocessing.get_context(self.start_method)
        cfg_snapshot = _config_snapshot()
        # Non-fork children can't inherit objects, so the user function and arguments are
        # serialized (cloudpickle supports functions defined in notebooks / __main__).
        pack = (lambda obj: obj) if self.start_method == "fork" else _dumps
        run_process_fn = pack(run_process_fn)

        init_ttp = cfg.mpc.provider == "TTP"
        ttp_count = 1 if init_ttp else 0
        total_ranks = world_size + ttp_count + evaluator_size

        mailbox = [ctx.Queue() for _ in range(total_ranks)]
        barrier = ctx.Barrier(world_size)
        self._result_queue = ctx.Queue()
        self.return_value = None
        if world_size == 2:
            shm_elems = InProcessCommunicator.SHM_SLOT_ELEMS
            shm_buffers = [ctx.RawArray(ctypes.c_int64, 4 * shm_elems) for _ in range(2)]
            peer_pipes = list(ctx.Pipe(duplex=True))
        else:
            shm_buffers = peer_pipes = None
        # Passed explicitly to every child (inherited state only exists under fork).
        shared = (shm_buffers, peer_pipes, cfg_snapshot)
        # Keep the IPC primitives alive in the parent: Process.start() drops its reference
        # to `args`, and garbage-collecting a Queue/Barrier unlinks its named semaphores
        # before 'spawn'/'forkserver' children have attached to them.
        self._ipc_refs = (mailbox, barrier, shared)

        self.processes = []
        self.ttp_process = None
        self.eval_processes = []

        for rank in range(world_size):
            new_args = copy.deepcopy(fn_args) if fn_args is not None else None
            if new_args is not None and getattr(new_args, "multi_gpu", False):
                new_args.device = f"cuda:{rank}"
            proc_device = (
                device
                if device is not None
                else (getattr(new_args, "device", None) if new_args is not None else None)
            )
            new_args = pack(new_args) if new_args is not None else None
            process_name = f"process {rank}"
            process = ctx.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(
                    rank,
                    world_size,
                    evaluator_size,
                    mailbox,
                    barrier,
                    run_process_fn,
                    new_args,
                    proc_device,
                    cfg_file,
                    self._result_queue if rank == 0 else None,
                    shared,
                ),
            )
            self.processes.append(process)

        if init_ttp:
            ttp_rank = world_size
            self.ttp_process = ctx.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(
                    ttp_rank,
                    world_size,
                    evaluator_size,
                    mailbox,
                    barrier,
                    TTPServer,
                    None,
                    None,
                    cfg_file,
                    None,
                    shared,
                ),
            )

        for i in range(evaluator_size):
            evaluator_rank = world_size + ttp_count + i
            process_name = f"evaluator process {evaluator_rank}"
            process = ctx.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(
                    evaluator_rank,
                    world_size,
                    evaluator_size,
                    mailbox,
                    barrier,
                    EvaluatorServer,
                    None,
                    "cpu",
                    cfg_file,
                    None,
                    shared,
                ),
            )
            self.eval_processes.append(process)

    @classmethod
    def _run_process(
        cls,
        rank,
        world_size,
        evaluator_size,
        mailbox,
        barrier,
        run_process_fn,
        fn_args,
        device=None,
        cfg_file=None,
        result_queue=None,
        shared=None,
    ):
        if isinstance(run_process_fn, bytes):
            run_process_fn = pickle.loads(run_process_fn)
        if isinstance(fn_args, bytes):
            fn_args = pickle.loads(fn_args)
        if shared is not None:
            shm_buffers, peer_pipes, cfg_snapshot = shared
            InProcessCommunicator.shm_buffers = shm_buffers
            InProcessCommunicator.peer_pipes = peer_pipes
            if cfg_snapshot is not None:
                # The parent already resolved cfg_file/mode; reuse its exact settings
                # (including programmatic overrides) instead of re-reading the file.
                cfg.set_config(cfg_snapshot)
                cfg_file = None
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        prix.init(
            rank,
            world_size,
            evaluator_size,
            mailbox,
            barrier,
            device=device,
            cfg_file=cfg_file,
        )
        logging.getLogger().setLevel(orig_logging_level)
        try:
            if fn_args is None:
                ret = run_process_fn()
            elif isinstance(fn_args, tuple) and len(fn_args) == 2 and fn_args[0] == "__prix_call_args__":
                call_args, call_kwargs = fn_args[1]
                ret = run_process_fn(*call_args, **call_kwargs)
            else:
                ret = run_process_fn(fn_args)
            if result_queue is not None and ret is not None:
                try:
                    result_queue.put(_to_host_return_value(ret))
                except Exception:
                    pass
        finally:
            if rank < world_size and prix.comm.is_initialized():
                prix.uninit()

    def start(self):
        if self.ttp_process is not None:
            self.ttp_process.start()
        for process in self.processes:
            process.start()
        for process in self.eval_processes:
            process.start()

    def join(self, timeout=None):
        """Waits for all MPC parties. If any party exits with a non-zero code,
        every remaining process is terminated immediately (so a crashed peer can
        never leave the others blocked forever) and a RuntimeError is raised."""
        deadline = None if timeout is None else time.monotonic() + timeout
        pending = list(self.processes)
        while pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready = multiprocessing.connection.wait([p.sentinel for p in pending], timeout=remaining)
            if not ready:
                self.terminate()
                raise TimeoutError(f"MPC parties did not finish within {timeout}s")
            for process in [p for p in pending if p.sentinel in ready]:
                process.join()
                pending.remove(process)
                if process.exitcode != 0:
                    self.terminate()
                    raise RuntimeError(
                        f"{process.name} exited with non-zero exit code {process.exitcode}"
                    )
        if self.ttp_process is not None:
            self.ttp_process.join(timeout=2.0)
        for process in self.eval_processes:
            process.join(timeout=2.0)
        self.terminate()
        try:
            if not self._result_queue.empty():
                self.return_value = self._result_queue.get_nowait()
        except Exception:
            pass
        return self.return_value

    def terminate(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        if self.ttp_process is not None and self.ttp_process.is_alive():
            self.ttp_process.terminate()
        for process in self.eval_processes:
            if process.is_alive():
                process.terminate()


def run(
    fn,
    *args,
    world_size=2,
    mode=None,
    evaluator_size=0,
    device=None,
    cfg_file=None,
    timeout=None,
    start_method=None,
    **kwargs,
):
    """
    Runs `fn(*args, **kwargs)` across `world_size` MPC parties and returns rank 0's result.

    `start_method` ('auto' by default, or 'fork' / 'spawn' / 'forkserver') controls how
    parties are started; see `_resolve_start_method`.
    """
    fn_args = ("__prix_call_args__", (args, kwargs)) if (args or kwargs) else None
    launcher = MultiProcessLauncher(
        world_size=world_size,
        evaluator_size=evaluator_size,
        run_process_fn=fn,
        fn_args=fn_args,
        cfg_file=cfg_file,
        mode=mode,
        device=device,
        start_method=start_method,
    )
    launcher.start()
    return launcher.join(timeout=timeout)


def run_multiprocess(
    world_size=2,
    mode=None,
    evaluator_size=0,
    device=None,
    cfg_file=None,
    start_method=None,
):
    """
    CrypTen-compatible decorator to run a function across `world_size` MPC parties:

        @prix.mpc.run_multiprocess(world_size=2, mode="shaft")
        def demo():
            x = prix.cryptensor([1.0, 2.0, 3.0])
            return (x * 2).get_plain_text()
    """
    # Support bare `@prix.mpc.run_multiprocess` without parentheses
    if callable(world_size):
        fn = world_size

        @functools.wraps(fn)
        def _bare_wrapper(*args, **kwargs):
            return run(fn, *args, world_size=2, **kwargs)

        return _bare_wrapper

    def _decorator(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            return run(
                fn,
                *args,
                world_size=world_size,
                mode=mode,
                evaluator_size=evaluator_size,
                device=device,
                cfg_file=cfg_file,
                start_method=start_method,
                **kwargs,
            )

        return _wrapper

    return _decorator


class MultiThreadLauncher:
    def __init__(
        self,
        world_size,
        evaluator_size,
        run_process_fn,
        fn_args=None,
    ):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        env["EVALUATOR_SIZE"] = str(evaluator_size)
        env["RENDEZVOUS"] = f"file:///tmp/prix-rendezvous-{uuid.uuid1()}"

        self.processes = []
        self.ttp_process = None
        self.eval_processes = []

        if threading.current_thread() != threading.main_thread():
            return
        self.threads = [
            threading.Thread(
                target=self._run, args=(run_process_fn, rank, world_size, fn_args)
            )
            for rank in range(world_size)
        ]
        for t in self.threads:
            t.start()

    def _run(self, test_fn, rank, world_size, fn_args):
        prix.init_thread(rank, world_size)
        test_fn(fn_args)
