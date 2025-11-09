"""
mpiexec -n 2 python3 secret_share_jax_mpi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes
from mpi4py import MPI
import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def share_secret(x, key):
    """Return two additive shares of x modulo MODULUS."""
    x1 = jax.random.randint(
        key,
        x.shape,
        dtypes.iinfo(jnp.int32).min,
        dtypes.iinfo(jnp.int32).max,
        dtype=jnp.int32,
    )
    x2 = x - x1
    return x1, x2


def main():
    key = jax.random.PRNGKey(0)

    if rank == 0:
        # Secret to share
        x = jnp.array([42, 99], dtype=jnp.int32)
        x1, x2 = share_secret(x, key)

        # Rank 0 keeps x1, sends x2
        _ = mpi4jax.send(x2, dest=1, comm=comm)
        print(f"[Rank {rank}] Sent share {x2}")
        my_share = x1

    elif rank == 1:
        # Receive x2
        my_share = mpi4jax.recv(jnp.zeros((2,), dtype=jnp.int32), source=0, comm=comm)
        print(f"[Rank {rank}] Received share {my_share}")

    # Each process has its own share
    print(f"[Rank {rank}] My share: {my_share}")

    # Optional: reconstruct using mpi4py (just for demo)
    gathered = comm.gather(np.array(my_share), root=0)
    if rank == 0:
        reconstructed = sum(gathered)
        print(f"[Rank {rank}] Reconstructed: {reconstructed}")


if __name__ == "__main__":
    main()
