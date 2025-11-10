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


class MultiProcessLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(
        self,
        world_size,
        evaluator_size,
        run_process_fn,
        fn_args=None,
    ):
        env = os.environ.copy()
        multiprocessing.set_start_method("fork")
        # if fn_args is None or 'device' not in fn_args:
        #     device = torch.device(device="cpu")
        # else:
        #     device = torch.device(fn_args.device)

        # Using multiple GPUs
        # if fn_args is not None and 'multi_gpu' in fn_args and fn_args.multi_gpu:
        #     assert (
        #         fn_args.world_size <= torch.cuda.device_count()
        #     ), f"Got {fn_args.world_size} parties, but only {torch.cuda.device_count()} GPUs found"

        mailbox = [multiprocessing.Queue() for _ in range(world_size + evaluator_size)]

        # This prevents one thread from running ahead of the others and doing
        # multiple puts that would show up in the get calls below
        barrier = multiprocessing.Barrier(world_size + evaluator_size)

        self.processes = []
        self.ttp_process = None  # Track the TTP process separately
        self.eval_processes = []
        for rank in range(world_size):
            if fn_args is not None and "multi_gpu" in fn_args and fn_args.multi_gpu:
                # device = torch.device(f"cuda:{rank}")
                new_args = copy.deepcopy(fn_args)
                # new_args.device = device
            else:
                new_args = fn_args
            device = None
            print(f"Running party {rank} in {device}")

            process_name = "process " + str(rank)
            process = multiprocessing.Process(
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
                ),
            )
            self.processes.append(process)

        # if cfg.mpc.provider == "TTP":
        #     if 'multi_gpu' in fn_args and fn_args.multi_gpu:
        #         if fn_args.world_size < torch.cuda.device_count():
        #             ttp_device = torch.device(f"cuda:{world_size}")
        #         else:
        #             # ttp_device = torch.device(f"cuda:0")
        #             ttp_device = torch.device(f"cpu")
        #     else:
        #         ttp_device = device

        # ttp_device = None
        # print(f'Running party {world_size} in {ttp_device}')
        # self.ttp_process = multiprocessing.Process(
        #     target=self.__class__._run_process,
        #     name="TTP",
        #     args=(
        #         world_size,
        #         world_size,
        #         evaluator_size,
        #         env,
        #         curl.mpc.provider.TTPServer,
        #         None,
        #         cfg_file,
        #         ttp_device,
        #     ),
        # )

        # for rank in range(evaluator_size):
        #     evaluator_rank = world_size + rank
        #     if self.ttp_process:
        #         evaluator_rank += 1
        #     if fn_args is not None and 'multi_gpu' in fn_args and fn_args.multi_gpu:
        #         device = torch.device(f"cpu")
        #         new_args = copy.deepcopy(fn_args)
        #         new_args.device = device
        #     else:
        #         new_args = fn_args
        #     print(f'Running party {evaluator_rank} in {device}')
        #
        #     process_name = "evaluator process " + str(evaluator_rank)
        #     process = multiprocessing.Process(
        #         target=self.__class__._run_process,
        #         name=process_name,
        #         args=(
        #             evaluator_rank,
        #             world_size,
        #             evaluator_size,
        #             env,
        #             curl.evaluator.EvaluatorServer,
        #             None,
        #             cfg_file,
        #             device
        #         ),
        #     )
        #     self.eval_processes.append(process)

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
    ):
        print(f"Creating process {rank}")
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        prix.init(rank, world_size, evaluator_size, mailbox, barrier, device=device)
        logging.getLogger().setLevel(orig_logging_level)
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

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
                print(f"{process.name} has non-zero exit code {process.exitcode}")

    def terminate(self):
        for process in self.processes:
            process.terminate()
        if self.ttp_process is not None and self.ttp_process.is_alive():
            self.ttp_process.terminate()
        for process in self.eval_processes:
            process.terminate()


class MultiThreadLauncher:

    # run_process_fn will be run in subprocesses.
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

        # Use random file so multiple jobs can be run simultaneously
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        env["RENDEZVOUS"] = INIT_METHOD

        self.processes = []
        self.ttp_process = None  # Track the TTP process separately
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
