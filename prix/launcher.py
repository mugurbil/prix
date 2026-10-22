#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import multiprocessing
import os
import threading
import uuid

import prix
from prix.config import cfg
from prix.evaluator import EvaluatorServer
from prix.provider import TTPServer


class MultiProcessLauncher:
    """Launches MPC parties, optional TTP server, and optional Fission Evaluator servers."""

    def __init__(
        self,
        world_size,
        evaluator_size=0,
        run_process_fn=None,
        fn_args=None,
        cfg_file=None,
    ):
        # Support both (world_size, evaluator_size, run_process_fn, fn_args)
        # and (world_size, run_process_fn, fn_args, cfg_file) signatures
        if callable(evaluator_size):
            if cfg_file is None:
                cfg_file = fn_args
            fn_args = run_process_fn
            run_process_fn = evaluator_size
            evaluator_size = getattr(fn_args, "evaluator_size", 0) if fn_args is not None else 0

        if cfg_file is not None:
            cfg.load_config(cfg_file)

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        ctx = multiprocessing.get_context("fork")

        init_ttp = cfg.mpc.provider == "TTP"
        ttp_count = 1 if init_ttp else 0
        total_ranks = world_size + ttp_count + evaluator_size

        mailbox = [ctx.Queue() for _ in range(total_ranks)]
        barrier = ctx.Barrier(world_size)

        self.processes = []
        self.ttp_process = None
        self.eval_processes = []

        for rank in range(world_size):
            new_args = copy.deepcopy(fn_args) if fn_args is not None else None
            if new_args is not None and getattr(new_args, "multi_gpu", False):
                new_args.device = f"cuda:{rank}"
            device = getattr(new_args, "device", None) if new_args is not None else None
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
                    device,
                    cfg_file,
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
    ):
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
                run_process_fn()
            else:
                run_process_fn(fn_args)
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

    def join(self):
        for process in self.processes:
            process.join()
            if process.exitcode != 0:
                self.terminate()
                raise RuntimeError(
                    f"{process.name} exited with non-zero exit code {process.exitcode}"
                )
        if self.ttp_process is not None:
            self.ttp_process.join(timeout=2.0)
        for process in self.eval_processes:
            process.join(timeout=2.0)

    def terminate(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        if self.ttp_process is not None and self.ttp_process.is_alive():
            self.ttp_process.terminate()
        for process in self.eval_processes:
            if process.is_alive():
                process.terminate()


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
