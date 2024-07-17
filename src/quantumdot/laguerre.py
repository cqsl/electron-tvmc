import jax
from jax import numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from typing import Any, Callable, Mapping, Optional
from functools import partial

@partial(jax.jit, static_argnames=('max_n',))
def genlaguerre_recurrence(
    n: Int[Array, "n"],
    alpha: Float[Array, ""],
    x: Float[Array, "m"],
    max_n: Int[Array, ""],
) -> Float[Array, "n m"]:
    """
    Computes the generalized Laguerre polynomial of degree n with parameter alpha at point x using the recurrence relation.

    Args:
        n: int, the degree of the generalized Laguerre polynomial.
        alpha: float, the parameter of the generalized Laguerre polynomial.
        x: float, the point at which to evaluate the polynomial.
        max_n: int, the maximum degree of n in the batch.

    Returns:
        The value of the generalized Laguerre polynomial of degree n with parameter alpha at point x.
    """
    # Initialize the array to store the generalized Laguerre polynomials for all degrees from 0 to max_n
    p = jnp.zeros((max_n + 1,) + x.shape, dtype=x.dtype)
    p = p.at[0].set(1.0)  # Set the 0th degree generalized Laguerre polynomial

    # Compute the generalized Laguerre polynomials for degrees 1 to max_n using the recurrence relation
    def body_fun(i, p):
        p_i = ((2 * i + alpha - 1 - x) * p[i - 1] - (i + alpha - 1) * p[i - 2]) / i
        return p.at[i].set(p_i)

    p = jax.lax.fori_loop(1, max_n + 1, body_fun, p)

    return p[n]

def eval_genlaguerre(
    n: Int[Array, "n"],
    alpha: Float[Array, ""],
    x: Float[Array, "m"],
    out: Float[Array, "n m"] = None,
) -> Float[Array, "n m"]:
    """
    Evaluates the generalized Laguerre polynomials of degrees specified in the input array n with parameter alpha at the points specified in the input array x.

    Args:
        n: array-like, the degrees of the generalized Laguerre polynomials.
        alpha: float, the parameter of the generalized Laguerre polynomials.
        x: array-like, the points at which to evaluate the polynomials.
        out: optional, an output array to store the results.

    Returns:
        An array containing the generalized Laguerre polynomial values of the specified degrees with parameter alpha at the specified points.
    """
    max_n = np.asarray(n).max()
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    
    if n.ndim == 0 and alpha.ndim == 0:
        p = genlaguerre_recurrence(n, alpha, x, max_n)
    elif n.ndim == 1 and alpha.shape == n.shape:
        p = jax.vmap(
            lambda ni, ai: genlaguerre_recurrence(ni, ai, x, max_n),
            out_axes=-1
        )(n, alpha)
    else:
        raise ValueError(f"shape of n and/or alpha not understood: {n.shape} and {alpha.shape}")

    # elif n.ndim == 1 and x.ndim == 1:
    #     p = jax.vmap(
    #         lambda ni: jax.vmap(
    #             lambda xi: genlaguerre_recurrence(ni, alpha, xi, max_n)
    #         )(x)
    #     )(n)
    #     p = jnp.diagonal(
    #         p
    #     )  # Get the diagonal elements to match the scipy.signal.eval_genlaguerre output
    # else:
    #     p = jax.vmap(
    #         lambda ni: jax.vmap(
    #             lambda xi: genlaguerre_recurrence(ni, alpha, xi, max_n)
    #         )(x)
    #     )(n)

    if out is not None:
        out = jnp.asarray(out)
        out = jnp.copy(p, out=out)
        return out
    else:
        return p
        # return jnp.squeeze(p)