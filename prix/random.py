import jax

from prix.encoder import DATATYPE


def random(seed, shape):
    """Random number generator"""
    return jax.random.randint(
        jax.random.PRNGKey(seed),
        shape,
        minval=jax.dtypes.iinfo(DATATYPE).min,
        maxval=jax.dtypes.iinfo(DATATYPE).max,
        dtype=DATATYPE,
    )
