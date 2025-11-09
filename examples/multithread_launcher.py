#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import threading
import uuid

import prix


class MultiThreadLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(
        self, world_size, evaluator_size, run_process_fn, fn_args=None, cfg_file=None
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
