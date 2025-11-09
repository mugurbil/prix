"""
mpiexec -n 2 python3 secret_share_jax_mpi.py
"""

import jax
import prix


def main():
    prix.init()

    rank = prix.comm.get().get_rank()

    if rank == 0:
        x = prix.array([42, 99], src=0, key=jax.random.PRNGKey(0))
    else:
        x = prix.array([0, 0], src=0)

    print(f"[Rank {rank}] My share: {x.share}")

    y = x.reconstruct(dst=0)

    print(f"[Rank {rank}] Reconstructed: {y}")

    # if rank == 0:
    #     # Secret to share
    #     x = jnp.array([42, 99], dtype=jnp.int32)
    #     x1, x2 = share_secret(x, key)
    #
    #     # Rank 0 keeps x1, sends x2
    #     _ = mpi4jax.send(x2, dest=1, comm=comm)
    #     print(f"[Rank {rank}] Sent share {x2}")
    #     my_share = x1
    #
    # elif rank == 1:
    #     # Receive x2
    #     my_share = mpi4jax.recv(jnp.zeros((2,), dtype=jnp.int32), source=0, comm=comm)
    #     print(f"[Rank {rank}] Received share {my_share}")
    #
    # # Each process has its own share
    # print(f"[Rank {rank}] My share: {my_share}")
    #
    # # Optional: reconstruct using mpi4py (just for demo)
    # gathered = comm.gather(np.array(my_share), root=0)
    # if rank == 0:
    #     reconstructed = sum(gathered)
    #     print(f"[Rank {rank}] Reconstructed: {reconstructed}")


if __name__ == "__main__":
    main()
