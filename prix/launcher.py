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
import secrets
import socket
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


def _free_ports(n):
    """Reserves n distinct free loopback TCP ports (released just before the children bind them)."""
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


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
        transport=None,
        tls=None,
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

        # Loading a preset for this run must not change the caller's global config
        # (otherwise e.g. run(..., mode="fission") would leave later runs in fission mode).
        caller_config = None
        if cfg_file is not None:
            caller_config = copy.deepcopy(object.__getattribute__(cfg, "config"))
            cfg.load_config(cfg_file)

        if evaluator_size == 0 and getattr(cfg.encoder, "truncation", None) == "fission":
            evaluator_size = 1

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        self.start_method = _resolve_start_method(start_method)
        ctx = multiprocessing.get_context(self.start_method)
        cfg_snapshot = _config_snapshot()
        init_ttp = cfg.mpc.provider == "TTP"
        if caller_config is not None:
            # Children receive cfg_snapshot explicitly, so the caller's config can be restored now.
            cfg.set_config(caller_config)
        # Non-fork children can't inherit objects, so the user function and arguments are
        # serialized (cloudpickle supports functions defined in notebooks / __main__).
        pack = (lambda obj: obj) if self.start_method == "fork" else _dumps
        run_process_fn = pack(run_process_fn)

        ttp_count = 1 if init_ttp else 0
        total_ranks = world_size + ttp_count + evaluator_size

        mailbox = [ctx.Queue() for _ in range(total_ranks)]
        barrier = ctx.Barrier(world_size)
        # Rank 0 sends its return value over a one-way pipe. The parent reads it while it is
        # still waiting on the processes (see join), so large results can't deadlock the way
        # a multiprocessing.Queue does when the parent only reads it after joining.
        self._result_recv, self._result_send = ctx.Pipe(duplex=False)
        self._result_received = False
        self.return_value = None
        transport = transport or os.environ.get("PRIX_TRANSPORT", "tcp")
        if transport not in ("shm", "tcp"):
            raise ValueError(f"transport must be 'shm' or 'tcp', got {transport!r}")
        self.transport = transport
        comm_kwargs = {}
        if transport == "tcp":
            # Every process (parties, TTP, evaluators) gets its own loopback port; the parties
            # then talk over direct TCP connections exactly as they would across hosts.
            comm_kwargs = {
                "transport": "tcp",
                "peers": [f"127.0.0.1:{p}" for p in _free_ports(total_ranks)],
                "session": secrets.token_hex(16),
                "tls": tls,
            }
        if world_size == 2 and transport == "shm":
            shm_elems = InProcessCommunicator.SHM_SLOT_ELEMS
            shm_buffers = [ctx.RawArray(ctypes.c_int64, 4 * shm_elems) for _ in range(2)]
            peer_pipes = list(ctx.Pipe(duplex=True))
        else:
            shm_buffers = peer_pipes = None
        # Passed explicitly to every child (inherited state only exists under fork).
        shared = (shm_buffers, peer_pipes, cfg_snapshot, comm_kwargs)
        # Keep the IPC primitives alive in the parent: Process.start() drops its reference
        # to `args`, and garbage-collecting a Queue/Barrier unlinks its named semaphores
        # before 'spawn'/'forkserver' children have attached to them.
        self._ipc_refs = (mailbox, barrier, shared, self._result_send)

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
                    self._result_send if rank == 0 else None,
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
        comm_kwargs = {}
        if shared is not None:
            shm_buffers, peer_pipes, cfg_snapshot, comm_kwargs = shared
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
            **comm_kwargs,
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
            if result_queue is not None:
                try:
                    result_queue.send(_to_host_return_value(ret))
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"PriX: could not send rank 0's return value to the caller: {e}")
                    result_queue.send(None)
                finally:
                    result_queue.close()
        finally:
            if prix.comm.is_initialized():
                prix.uninit()

    def start(self):
        if self.ttp_process is not None:
            self.ttp_process.start()
        for process in self.processes:
            process.start()
        for process in self.eval_processes:
            process.start()
        # Drop the parent's copy of the send end so a crashed rank 0 shows up as EOF.
        self._result_send.close()

    def _receive_result(self):
        try:
            self.return_value = self._result_recv.recv()
        except (EOFError, OSError):
            pass
        self._result_received = True

    def join(self, timeout=None):
        """Waits for all MPC parties. If any party exits with a non-zero code,
        every remaining process is terminated immediately (so a crashed peer can
        never leave the others blocked forever) and a RuntimeError is raised."""
        deadline = None if timeout is None else time.monotonic() + timeout
        pending = list(self.processes)
        helpers = [p for p in [self.ttp_process] + self.eval_processes if p is not None]
        while pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            waitables = [p.sentinel for p in pending] + [h.sentinel for h in helpers if h.is_alive()]
            if not self._result_received:
                waitables.append(self._result_recv)
            ready = multiprocessing.connection.wait(waitables, timeout=remaining)
            if not ready:
                self.terminate()
                raise TimeoutError(f"MPC parties did not finish within {timeout}s")
            if self._result_recv in ready:
                self._receive_result()
            for helper in [h for h in helpers if h.sentinel in ready]:
                helper.join()
                if helper.exitcode not in (0, None):
                    self.terminate()
                    raise RuntimeError(f"{helper.name} exited with non-zero exit code {helper.exitcode}")
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
        if not self._result_received and self._result_recv.poll(0):
            self._receive_result()
        self.terminate()
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
    transport=None,
    tls=None,
    **kwargs,
):
    """
    Runs `fn(*args, **kwargs)` across `world_size` MPC parties and returns rank 0's result.

    `start_method` ('auto' by default, or 'fork' / 'spawn' / 'forkserver') controls how
    parties are started; see `_resolve_start_method`.

    `transport` selects how the local processes talk: "tcp" (default; direct TCP connections
    over loopback, the same code path as a multi-host deployment; `tls` optionally enables
    mutual TLS) or "shm" (shared memory, a single-host simulation shortcut). The default can
    also be set with the PRIX_TRANSPORT environment variable.
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
        transport=transport,
        tls=tls,
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
    transport=None,
    tls=None,
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
                transport=transport,
                tls=tls,
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
