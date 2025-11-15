#!/usr/bin/env python3

"""
python tests/test_process.py
"""

import argparse
import logging

import jax.numpy as jnp

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


def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    logging.getLogger().setLevel(level)

    rank = prix.comm.get().get_rank()

    # Test Sharing
    inner = None
    if rank == 0:
        inner = [1.5, 2.75]
    arr = prix.array(inner, precision=4, src=0)
    print(f"[Rank {rank}] My share: {arr.share}")
    reconstructed = arr.reconstruct(dst=0)
    print(f"[Rank {rank}] Reconstructed: {reconstructed}")
    if rank == 0:
        assert jnp.array_equal(
            reconstructed, jnp.array(inner)
        ), f"Reconstructed {reconstructed} != {inner}"
    elif rank == 1:
        assert reconstructed == inner, f"None {reconstructed} != {inner}"

    x_inner = jnp.array([[0, 1], [1, 0]])
    x = prix.array(x_inner, precision=8, src=0)
    y_inner = jnp.array([[21.25, 0], [1, 11.125]])
    y = prix.array(y_inner, precision=8, src=1)

    print("Test Private Recon")
    left, right = x.reconstruct(), x_inner
    assert jnp.array_equal(left, right), f"Reconstruct {left} != {right}"

    print("Test Private Add")
    left, right = (x + y).reconstruct(), x_inner + y_inner
    assert jnp.array_equal(left, right), f"Add {left} != {right}"

    print("Test Private Neg")
    left, right = (-y).reconstruct(), -y_inner
    assert jnp.array_equal(left, right), f"Neg {left} != {right}"

    print("Test Private Sub")
    left, right = (x - y).reconstruct(), x_inner - y_inner
    assert jnp.array_equal(left, right), f"Sub {left} != {right}"

    print("Test Private MatMul")
    left, right = (x @ y).reconstruct(), x_inner @ y_inner
    assert jnp.array_equal(left, right), f"MatMul {left} != {right}"

    print("Test Private Mul")
    left, right = (x * y).reconstruct(), x_inner * y_inner
    assert jnp.array_equal(left, right), f"Mul {left} != {right}"

    z = jnp.array([2, 3])

    print("Test Public MatMul")
    left, right = (x @ z).reconstruct(), x_inner @ z
    assert jnp.array_equal(left, right), f"Public MatMul {left} != {right}"

    print("Test Public Mul")
    left, right = (x * z).reconstruct(), x_inner * z
    assert jnp.array_equal(left, right), f"Public MatMul {left} != {right}"

    print("Test Right Public MatMul")
    left, right = (z @ x).reconstruct(), z @ x_inner
    assert jnp.array_equal(left, right), f"Public MatMul {left} != {right}"

    print("Test Right Public Mul")
    left, right = (z * x).reconstruct(), z * x_inner
    assert jnp.array_equal(left, right), f"Public MatMul {left} != {right}"

    print("Done")


def main():
    args = get_args()

    print(f"{args.world_size} parties")

    launcher = MultiProcessLauncher(
        args.world_size, args.evaluator_size, _run_experiment, args
    )
    launcher.start()
    launcher.join()
    launcher.terminate()


if __name__ == "__main__":
    main()
