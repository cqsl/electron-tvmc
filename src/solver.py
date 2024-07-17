# import netket as nk
import jax.numpy as jnp
from functools import partial
import jax
import numpy as np
from src.utils import real_dtype
from netket.jax import tree_ravel

def smooth_svd(Aobj, b, acond=1e-4, rcond=1e-2, exponent=6, x0=None):
    """
    Solve the linear system using Singular Value Decomposition.
    The diagonal shift on the matrix should be 0.
    Internally uses {ref}`jax.numpy.linalg.lstsq`.
    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rcond: The condition number
        
    from Medvidovic and Sels
    """
    del x0

    A = Aobj.to_dense()

    b, unravel = tree_ravel(b)

    s2, V = jnp.linalg.eigh(A)
    del A # memory saving
    
    b_tilde = V.T.conj() @ b

    svd_reg = _default_reg_fn(s2, rcond=rcond, acond=acond, exponent=exponent)

    cutoff = 10 * jnp.finfo(s2.dtype).eps
    s2_safe = jnp.maximum(s2, cutoff)
    reg_inv = svd_reg / s2_safe

    x = V @ (reg_inv * b_tilde)
    effective_rank = jnp.sum(svd_reg)

    info = {
        "effective_rank" :  effective_rank,
        "svd_reg" : svd_reg,
        "s2" : s2,
        "max_s2" : jnp.max(s2)
    }
        
    del V # memory saving

    return unravel(x), info


def _default_reg_fn(x, rcond, acond, exponent):

    cutoff = jnp.finfo(real_dtype(x.dtype)).eps

    if acond is not None:
        cutoff = jnp.maximum(cutoff, acond)

    cutoff = jnp.maximum(cutoff, rcond * jnp.max(x))

    return 1 / (1 + (cutoff / x) ** exponent)