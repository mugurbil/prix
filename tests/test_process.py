#!/usr/bin/env python3

"""
python test_process.py --multiprocess
"""

import argparse
import logging

import prix

from prix.launcher import MultiProcessLauncher


def get_args():
    parser = argparse.ArgumentParser(description="Prix LLM Inference")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--evaluator_size",
        "-es",
        type=int,
        default=0,
        help="The number of eval parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--multiprocess",
        default=False,
        action="store_true",
        help="Run example in multiprocess mode",
    )
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        default="cpu",
        help="the device to run the benchmarks",
    )
    parser.add_argument(
        "--multi-gpu",
        "-mg",
        required=False,
        default=False,
        action="store_true",
        help="use different gpu for each party. Will override --device if selected",
    )
    args = parser.parse_args()
    return args


def run_shuffle(device=None):
    rank = prix.comm.get().get_rank()

    if rank == 0:
        x = [42.5, 99.75]
    else:
        x = None

    x = prix.array(x, precision=16, src=0)
    y = prix.array([21.25, 11.125], precision=8, src=1)

    z = x - y

    print(f"[Rank {rank}] My share: {x.share} {y.share} {z.share}")

    recon = z.reconstruct(dst=0)

    print(f"[Rank {rank}] Reconstructed: {recon}")


def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    # if "RANK" in os.environ and os.environ["RANK"] != "0":
    #     level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    run_shuffle()
    print("Done")


def main():
    args = get_args()

    print(f"{args.world_size} parties")

    if args.multiprocess:
        launcher = MultiProcessLauncher(
            args.world_size, args.evaluator_size, _run_experiment, args
        )
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)


if __name__ == "__main__":
    main()
