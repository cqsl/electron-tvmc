import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import netket.jax as nkjax
from netket.utils import mpi
from netket.stats import mean as mpi_mean
from netket.stats import statistics, Stats
import scipy
import matplotlib.pyplot as plt
import itertools


@partial(jax.jit, static_argnames=("apply_fun", "chunk_size", "hilb_size"))
def _log_value_chunked(apply_fun, variables, x, chunk_size=None, hilb_size=None):
    def _log_psi(s):
        return apply_fun(variables, s)

    if hilb_size is not None:
        x = x.reshape(-1, hilb_size)
    return nkjax.apply_chunked(_log_psi, chunk_size=chunk_size)(x)

@jax.jit
def permute_samples(key, x):
    assert x.ndim == 3
    idxs = jnp.arange(x.shape[0])
    idxs = jax.random.permutation(key, idxs)
    return x[idxs, ...]

@jax.jit
def permute_particles(key, x):
    assert x.ndim == 3
    n_samples = x.shape[0]
    N = x.shape[1]
    keys = jax.random.split(key, n_samples)
    keys = jnp.asarray(keys)
    part_idxs = jax.vmap(lambda k: jax.random.permutation(k, jnp.arange(N)))(keys)
    return jax.vmap(lambda xi, idxi: xi[idxi, :])(x, part_idxs)


@partial(
    jax.jit, static_argnames=("apply_fun", "chunk_size", "return_aux", "sum_all_combs")
)
def _compute_full_g2_terms(
    apply_fun, variables, R, chunk_size=None, return_aux=False, sum_all_combs=True # symmetrization improves greatly the statistics 
): 
    assert R.ndim == 3
    sdim = R.shape[-1]
    N = R.shape[-2]
    hilb_size = N * sdim
    n_samples = R.shape[0]

    log_psi = lambda _x: _log_value_chunked(
        apply_fun, variables, _x, chunk_size=chunk_size, hilb_size=hilb_size
    )
    lv_R = log_psi(R)

    term1_loc = jnp.full((n_samples,), N * (N - 1), dtype=jnp.float64)
    term1 = statistics(term1_loc)

    term2x_loc = jnp.full((n_samples,), N, dtype=jnp.float64)
    term2y_loc = jnp.full((n_samples,), N, dtype=jnp.float64)
    term2_loc = term2x_loc * term2y_loc
    term2 = statistics(term2_loc)

    assert R.shape[0] % 2 == 0, f"got R shape={R.shape}"
    R, Rp = jnp.split(R, 2, axis=0)
    lv_R, lv_Rp = jnp.split(lv_R, 2, axis=0)
    if not sum_all_combs:
        log_term3_a = log_psi(R.copy().at[:, 0, :].set(Rp[:, 0, :])) - lv_R
        log_term3_b = log_psi(Rp.copy().at[:, 0, :].set(R[:, 0, :])) - lv_Rp
        log_term3 = log_term3_a + log_term3_b
        term3_loc = jnp.exp(log_term3) * N**2
    else:
        pair_idxs = np.array(list(itertools.product(range(N), repeat=2)))
        log_term3_a = (
            jax.vmap(lambda i, j: log_psi(R.copy().at[:, i, :].set(Rp[:, j, :])))(
                pair_idxs[:, 0], pair_idxs[:, 1]
            )
            - lv_R # broadcast
        )
        # log_term3_a = nkjax.logsumexp_cplx(log_term3_a, axis=0) # over the vmap
        log_term3_b = (
            jax.vmap(lambda i, j: log_psi(Rp.copy().at[:, j, :].set(R[:, i, :])))(
                pair_idxs[:, 0], pair_idxs[:, 1]
            )
            - lv_Rp
        )
        log_term3 = log_term3_a + log_term3_b # first multiply per sample and per pairing
        log_term3 = nkjax.logsumexp_cplx(log_term3, axis=0) # sum over the permutations
        # log_term3_b = nkjax.logsumexp_cplx(log_term3_b, axis=0)
        # log_term3 = log_term3_a + log_term3_b
        term3_loc = jnp.exp(log_term3)
    term3 = statistics(term3_loc)

    corr_add = term1.mean - term2.mean + term3.mean
    corr_corr = term1.mean * N / (N - 1) - term2.mean + term3.mean

    corr_loc = term1_loc - term2_loc + jnp.repeat(term3_loc, 2)  # make sizes compatible
    corr = statistics(corr_loc)

    # print(term1.shape, term2.shape, term3.shape)

    if return_aux:
        aux = {
            "corr": corr_add,
            "terms": [term1.mean, term2.mean, term3.mean],
            "terms_sigmas": [
                term1.error_of_mean,
                term2.error_of_mean,
                term3.error_of_mean,
            ],
            "densityX": statistics(term2x_loc).mean,
            "densityY": statistics(term2y_loc).mean,
            "corr_corr": corr_corr,
        }
        return corr, aux
    else:
        return corr


@jax.jit
def permute_samples(key, x):
    assert x.ndim == 3
    idxs = jnp.arange(x.shape[0])
    idxs = jax.random.permutation(key, idxs)
    return x[idxs, ...]


def compute_full_g2(vs, return_aux=False, permute=True, sum_all_combs=True):
    """Compute the g2(x, y) for a bunch of Xs and Ys (which are pairs)

    Args:
        Xs = [number of points, sdim]
        Ys = [number of points, sdim]
        return_aux = gives auxiliar info about the 3 contributing terms
        permute = whether to permute the particle indices in each sample to improve statistics
    """
    sdim = len(vs.hilbert.extent)
    N = vs.hilbert.n_particles

    R = vs.samples
    R = R.reshape(-1, N, sdim)
    if permute:
        # due to indistinguishability, this should not matter
        key = vs.sampler_state.rng
        key1, key2 = jax.random.split(key, 2)
        R = permute_particles(key1, R)
        # order of samples should not matter
        R = permute_samples(key2, R)

    _local_g2_psi = partial(
        _compute_full_g2_terms,
        vs._apply_fun,
        vs.variables,
        sum_all_combs=sum_all_combs,
        chunk_size=vs.chunk_size,
        return_aux=return_aux,
    )
    out = _local_g2_psi(R)
    return out    