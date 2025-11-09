#!/usr/bin/env python3

"""
python test_process.py --multiprocess
"""

import argparse
import logging

import prix
import jax

from examples.multiprocess_launcher import MultiProcessLauncher


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


def get_config(args):
    cfg_file = prix.cfg.get_default_config_path()
    if args.evaluator_size:
        logging.info("Using Fission config")
        cfg_file = cfg_file.replace("default", "fission")
    return cfg_file


def run_shuffle(cfg_file, device=None):
    # prix.init(cfg_file, device=device)

    rank = prix.comm.get().get_rank()

    if rank == 0:
        x = prix.array([42, 99], src=0)
    else:
        x = prix.array([0, 0], src=0)

    print(f"[Rank {rank}] My share: {x.share}")

    y = x.reconstruct(dst=0)

    print(f"[Rank {rank}] Reconstructed: {y}")


def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    # if "RANK" in os.environ and os.environ["RANK"] != "0":
    #     level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    cfg_file = get_config(args)
    run_shuffle(cfg_file)
    print("Done")


def main():
    args = get_args()
    cfg_file = get_config(args)
    prix.cfg.load_config(cfg_file)

    print(f"{args.world_size} parties")

    if args.multiprocess:
        launcher = MultiProcessLauncher(
            args.world_size, args.evaluator_size, _run_experiment, args, cfg_file
        )
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)


if __name__ == "__main__":
    main()
