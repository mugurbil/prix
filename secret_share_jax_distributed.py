import os
import jax
import jax.numpy as jnp
from jax import dtypes
from jax.experimental import multihost_utils
import jax.distributed as distributed

jax.config.update("jax_enable_x64", True)


def share_secret(x):
    key = jax.random.PRNGKey(0)
    x1 = jax.random.randint(
        key,
        x.shape,
        minval=dtypes.iinfo(jnp.int64).min,
        maxval=dtypes.iinfo(jnp.int64).max,
        dtype=jnp.int64,
    )
    x2 = x - x1
    return x1, x2


def main():
    # --- Distributed Initialization ---
    coordinator_address = os.environ["JAX_COORDINATOR_ADDRESS"]
    num_processes = int(os.environ["JAX_PROCESS_COUNT"])
    process_id = int(os.environ["JAX_PROCESS_INDEX"])

    distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )

    print(f"[Process {process_id}] Initialized.")

    # Barrier: ensure both processes are ready
    multihost_utils.sync_global_devices("sync-before-broadcast")

    print("Process index:", jax.process_index())
    print("Process count:", jax.process_count())
    print("Devices:", jax.devices())

    if process_id == 0:
        x = jnp.array([[42, 99], [123, 456]], dtype=jnp.int64)
        x1, x2 = share_secret(x)
        print(f"[Process 0] Original tensor:\n{x}")
        print(f"[Process 0] My share x1:\n{x1}")

        multihost_utils.broadcast_one_to_all(x2, is_source=True)
        print(f"[Process 0] Other share x2:\n{x2}")
        print(f"[Process 0] Decrpyted:\n{x1 + x2}")
        print(f"[Process {process_id}] Sent share x2.")
    else:
        # Dummy input must match shape/dtype
        dummy = jnp.zeros((2, 2), dtype=jnp.int64)
        shared_x2 = multihost_utils.broadcast_one_to_all(dummy, is_source=False)
        print(f"[Process {process_id}] Received share x2:\n{shared_x2}")

    # Barrier: sync after broadcast
    multihost_utils.sync_global_devices("sync-after-broadcast")


if __name__ == "__main__":
    main()
