import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import functools
from typing import (Optional, Sequence)
# from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from jax.scipy.special import gammaln
from netket.utils import mpi
import tqdm
import flax

import jax
import numpy as np
import jax.numpy as jnp
from netket.utils.mpi import mpi_sum, n_nodes, mpi_allgather_jax, mpi, MPI_jax_comm, mpi_bcast_jax
from netket.jax import mpi_split
import netket as nk
import json
import copy


def pytree_array_string(tree, indent=4):
    def _to_str(x):
        if x is None:
            return "(NONE!)"
        else:
            return f"{x.shape}=({x.size}) -> {x.dtype}"
    td = jax.tree_map(_to_str, tree)
    return json.dumps(td, indent=indent)

@jax.jit
def get_spin_spin_matrix(si):
    return si[:, None] * si[None, :]

@jax.jit
def get_el_ion_distance_matrix(r_el, R_ion):
    """
    Args:
        r_el: shape [N_batch x n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [N_batch x n_el x N_ion x 3]
        dist: shape [N_batch x n_el x N_ion]
    """
    diff = r_el[..., None, :] - R_ion[..., None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist

@jax.jit
def get_full_distance_matrix(r_el):
    """
    Args:
        r_el: shape [n_el x 3]
    Returns:
    """
    diff = jnp.expand_dims(r_el, -2) - jnp.expand_dims(r_el, -3)
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist

@jax.jit
def dists_from_diffs_matrix(r_el_diff):
    n_el = r_el_diff.shape[-2]
    diff_padded = r_el_diff + jnp.eye(n_el)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el))
    return dist


@jax.jit
def get_distance_matrix(r_el):  # stable!
    """
    Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)
    Args:
        r_el: [batch_dims x n_electrons x 3]
    Returns:
        tuple: differences [batch_dims x n_el x n_el x 3], distances [batch_dims x n_el x n_el]
    """
    diff = r_el[..., :, None, :] - r_el[..., None, :, :]
    dist = dists_from_diffs_matrix(diff)
    return diff, dist


@partial(jax.jit, static_argnums=(1, 2))
def pick_triu(x, has_aux=False, k=1):
    n = x.shape[-2]
    idxs = jnp.triu_indices(n, k=k)
    if has_aux:
        return x[..., idxs[0], idxs[1], :]
    else:
        return x[..., idxs[0], idxs[1]]

def n_square_from_triu(n_flat):
    return int(np.rint((1 + np.sqrt(1 + 8*n_flat))/2))

# def logsumexp(*args, **kwargs):
#     return jax.scipy.special.logsumexp(*args, **kwargs)
def logsumexp(*args, **kwargs):
    return nk.jax.logsumexp_cplx(*args, **kwargs)

@jax.jit
def slogdet(x):
    """Computes sign and log of determinants of matrices.
    This is a jnp.linalg.slogdet with a special (fast) path for small matrices.
    Args:
        x: square matrix.
    Returns:
        sign, (natural) logarithm of the determinant of x.
    """
    """
    if x.shape[-1] == 1:
        sign = jnp.sign(x[..., 0, 0])
        logdet = jnp.log(jnp.abs(x[..., 0, 0]))
    else:
    """
    raise Exception("NOT FOR REAL!")
    sign, logdet = jnp.linalg.slogdet(x.astype(complex))
    return sign, logdet


@jax.jit
def vmap_logdet_matmul(
    xs: Sequence[jnp.ndarray],
    w: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    def fn(mat): 
        return logdet_matmul(mat, w=w)
    return jax.vmap(fn)(xs)


@jax.jit
def logdet_matmul(
    xs: Sequence[jnp.ndarray],
    w: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    raise Exception("NOT FOR IMAGINARY!")
    """Combines determinants and takes dot product with weights in log-domain.
    We use the log-sum-exp trick to reduce numerical instabilities.
    Args:
        xs: FermiNet orbitals in each determinant. Either of length 1 with shape
            (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
            (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
            determinants are factorised into block-diagonals for each spin channel).
        w: weight of each determinant. If none, a uniform weight is assumed.
    Returns:
        sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
        determinant (or product of the i-th determinant in each spin channel, if
        full_det is not used).
    """
    if isinstance(xs, jnp.ndarray):
        xs = [xs]

    # Special case to avoid taking log(0) if any matrix is of size 1x1.
    # We can avoid this by not going into the log domain and skipping the
    # log-sum-exp trick.
    det1 = functools.reduce(
        lambda a, b: a * b,
        [x.reshape(-1) for x in xs if x.shape[-1] == 1],
        1
    )
    # Compute the logdet for all matrices larger than 1x1
    sign_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogdet(x) for x in xs if x.shape[-1] > 1],
        (1, 0)
    )

    res, sign = logsumexp(jnp.log(det1) + logdet, b=w, return_sign=True)
    return sign*sign_in, res


@partial(jax.jit, static_argnums=1)
def remove_diag(A, has_aux_axis=False):
    n = A.shape[-2]
    mask = np.eye(n, dtype=bool)
    if has_aux_axis:
        assert A.shape[-3] == n
        batch_shape = A.shape[:-3]
        aux_size = A.shape[-1]
        return A[...,~mask,:].reshape(*batch_shape, n, n-1, aux_size)
    else:
        assert A.shape[-1] == n
        batch_shape = A.shape[:-2]
        return A[...,~mask].reshape(*batch_shape, n, n-1)


def convert_to_useful(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.complexfloating):
            return {
                "real": np.ascontiguousarray(obj.real),
                "imag": np.ascontiguousarray(obj.imag),
            }
    elif isinstance(obj, jax.numpy.ndarray):
        return np.ascontiguousarray(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}

    raise TypeError

def tqdm_mpi(x):
    if mpi.node_number == 0:
        return tqdm.tqdm(x)
    else:
        return x

def get_max_conn(op, verbose=False, change=True, safety_factor=1.1, chunk_size=None, n_samples=1024):
    hi = op.hilbert
    # the following also calls setup

    max_conn = 0
    samples = hi.random_state(key=jax.random.PRNGKey(0), size=(n_samples,))
    if chunk_size is None:
        chunk_size = n_samples
    samples = samples.reshape(-1, chunk_size, hi.size)

    n_conn_list = []
    for x in tqdm_mpi(samples):
        _, mels = op.get_conn_padded(x)
        mels = mels.reshape(-1, mels.shape[-1])
        n_conn = (1-np.isclose(mels, 0)).sum(axis=-1)
        _max_conn = np.max(n_conn)
        max_conn = max(_max_conn, max_conn)
        n_conn_list.append(n_conn)
    n_conn_list = np.array(n_conn_list).ravel()

    if verbose:
        print("original max conn = ", op.max_conn_size)
        print("max conn = ", max_conn)
        plt.scatter(np.arange(n_conn_list.shape[0]), n_conn_list)
        plt.show()

    if change:
        assert safety_factor >= 1
        assert op._initialized
        op._max_conn_size = int(max_conn*safety_factor)
        op._max_conn_size
    return max_conn


def factorial2(n):
    n = jnp.asarray(n)
    gamma = jnp.exp(gammaln(n / 2 + 1))
    factor = jnp.where(
        n % 2, jnp.power(2, n / 2 + 0.5) /
        jnp.sqrt(jnp.pi), jnp.power(2, n / 2)
    )
    return factor * gamma


def print_mpi(*args, **kwargs):
    if mpi.node_number == 0:
        print(*args, **kwargs)

# @jax.jit
def is_complex_dtype(dtype) -> bool:
    return jnp.issubdtype(dtype, jnp.complexfloating)

def real_dtype(dtype):

    if is_complex_dtype(dtype):
        if dtype == np.dtype("complex64"):
            return np.dtype("float32")
        elif dtype == np.dtype("complex128"):
            return np.dtype("float64")
        else:
            raise TypeError(f"Unknown complex dtype {dtype}.")
    else:
        return np.dtype(dtype)
    
def complex_dtype(dtype):
    if is_complex_dtype(dtype):
        return np.dtype(dtype)
    else:
        if dtype == np.dtype("float32"):
            return np.dtype("complex64")
        elif dtype == np.dtype("float64"):
            return np.dtype("complex128")
        else:
            raise TypeError(f"Unknown complex dtype {dtype}.")
    
def nm_to_au(x):
    return x*18.8972598858

def Wcm2_to_au(x):
    return x/(3.50944758e16)

def fs_to_au(x):
    return x*41.341374575751

def ang_to_au(x):
    return x*1.8897259886


@jax.jit
def safe_power(x, ls):
    x, ls = jnp.broadcast_arrays(x, ls)
    # instability 1: x**0 --> 1. --> 1.0**0
    ls_zero = ls == 0
    x_safe = jnp.where(ls_zero, 1., x)
    return jnp.power(x_safe, ls)

def print_discrete_samples(x, n_print=25):
    print_mpi()
    x = x.reshape(-1, x.shape[-1]).astype(int)
    print_mpi("n_samples before unique:", x.shape[0])
    xu, c = np.unique(x, axis=0, return_counts=True)
    print_mpi("n_samples after unique:", xu.shape[0])
    idx = np.argsort(c)[::-1]
    # take only first few samples
    xu = xu[idx,:][:n_print,:]
    c = c[idx][:n_print]
    for xi, ci in zip(xu, c):
        xi_up, xi_dn = np.split(xi, 2)
        print_mpi("".join(xi_up.astype(str).tolist()) + "|" + "".join(xi_dn.astype(str).tolist()), " --> ", ci)
    print_mpi(f"... [only first {n_print} shown]")
    print_mpi(flush=True)
    
    
def tree_size_cumsum(tree):
    p = jax.tree_util.tree_map(np.size, tree)
    p = flax.traverse_util.flatten_dict(p, keep_empty_nodes=False, is_leaf=None, sep="/")
    k = list(p.keys())
    v = list(p.values())
    v = np.cumsum(v)
    v = list(map(int, v))
    p = dict(zip(k,v))
    return p

def mpi_same(key, *, root=0, comm=MPI_jax_comm):
    keys = [key for _ in range(mpi.n_nodes)]
    keys = jax.tree_map(lambda k: mpi_bcast_jax(k, root=root, comm=comm)[0], keys)
    return keys[mpi.rank]

def shuffle_mpi(key, a, node_number): # the key needs to be the same (!!!)
    key = mpi_same(key) # make sure we shuffle the same
    b = mpi_allgather_jax(a)[0]
    if b.ndim == a.ndim:
        b = a[None,...]
    idxs = jax.random.permutation(key, np.arange(b.shape[0]))
    idx = idxs[node_number]
    c = b[idx,...]
    return c

def shuffle_along_axis(key, a, axis=0):
    return jax.random.permutation(key, a, axis=axis, independent=False)

def mpi_shuffle_along_axis(key, a, node_number, axis=0):
    a = np.array(a)
    keys = jax.random.split(key, a.shape[axis])
    out = []
    for i in range(a.shape[axis]):
        b = np.take(a, i, axis).copy()
        c = shuffle_mpi(keys[i], b, node_number) # different key for each element on the axis
        out.append(c.copy()) 
    out = np.stack(out, axis=axis)
    return out

def copy_frozen_dict(frozen_dict):
    return type(frozen_dict)({**frozen_dict})

def copy_variational_state(vs, n_hot=0, copy_samples=True, **kwargs):
    # warning: does not use the initial sampler state values
    apply_fun = kwargs.get("apply_fun", None)
    init_fun = kwargs.get("init_fun", None)
    variables = kwargs.get("variables", None)
    # model = vs.model
    # apply = model.apply if apply_fun is None else apply_fun
    # init = model.init if init_fun is None else init_fun
    apply = vs._apply_fun if apply_fun is None else apply_fun
    init = vs._init_fun if init_fun is None else init_fun
    if variables is not None:
        variables = copy_frozen_dict(variables)

    if isinstance(vs, nk.vqs.MCState) or hasattr(vs, "samples"):
        n_samples = kwargs.get("n_samples", vs.n_samples)
        copy_vs = nk.vqs.MCState(
            vs.sampler,
            apply_fun=apply,
            init_fun=init,
            n_samples=n_samples,
            # model=model,
            chunk_size=vs.chunk_size,
            n_discard_per_chain=vs.n_discard_per_chain,
            variables=variables,
        )
        if variables is None:
            copy_vs.parameters = copy_frozen_dict(vs.parameters)
        if copy_samples and hasattr(vs.sampler_state, "σ") and n_samples == vs.n_samples:
            copy_vs.sampler_state = copy_vs.sampler_state.replace(σ=vs.sampler_state.σ)
        for _ in range(n_hot):
            copy_vs.reset()
            copy_vs.samples
        return copy_vs
    elif isinstance(vs, nk.vqs.FullSumState) or hasattr(vs, "_all_states"):
        copy_vs = nk.vqs.FullSumState(
            vs.hilbert, 
            apply_fun=apply,
            init_fun=init,
            variables=variables,
            # model=model,
        )
        if variables is None:
            copy_vs.parameters = copy_frozen_dict(vs.parameters)
        return copy_vs
    else:
        raise NotImplementedError(f"cannot copy this kind of state {type(vs)}")
    
    

def burn_in(vs, n=10):
    for _ in range(n):
        vs.reset()
        vs.samples
    return vs

@jax.jit
def add_noise_to_param_dict(key, d, stddev=None):
    if stddev is None:
        stddev = 0
    leaves, tree_def = jax.tree_util.tree_flatten(d)
    n_leaves = len(leaves)
    keys = jax.random.split(key, n_leaves)
    key_tree = jax.tree_util.tree_unflatten(tree_def, keys)
    return jax.tree_util.tree_map(lambda x, k: x + stddev*jax.random.normal(k, shape=x.shape), d, key_tree)
    

@jax.jit 
def safe_log(x):
    is_finite = x != 0
    x = jnp.where(is_finite, x, 1)
    inf_add = jnp.where(is_finite, 0, -jnp.inf)
    if not jnp.iscomplexobj(x): # save an addition or casting
        x = x + 0j
    return jnp.log(x+0j) + inf_add    

@partial(jax.jit, static_argnames=("axis", "keepdims"))
def mpi_logmeanexp_jax(a, axis=None, keepdims=False):
    """ Compute Log[Mean[Exp[a]]]"""
    # subtract logmax for better numerical stability
    a_max = mpi_max_jax(jnp.max(a.real, axis=axis, keepdims=True))[0]
    a_max = jax.lax.stop_gradient(a_max)
    a_shift = a - a_max
    exp_a = jnp.exp(a_shift)
    exp_a_mean = mpi_mean_jax(jnp.mean(exp_a, axis=axis, keepdims=True))[0]
    log_mean = safe_log(exp_a_mean) + a_max
    if not keepdims:
        log_mean = log_mean.squeeze(axis=axis)
    return log_mean

@partial(jax.jit, static_argnames=("axis", "keepdims"))
def mpi_logsumexp_jax(a, axis=None, keepdims=False):
    """ Compute Log[Mean[Exp[a]]]"""
    # subtract logmax for better numerical stability
    a_max = mpi_max_jax(jnp.max(a.real, axis=axis, keepdims=True))[0]
    a_max = jax.lax.stop_gradient(a_max)
    a_shift = a - a_max
    exp_a = jnp.exp(a_shift)
    exp_a_sum = mpi_sum_jax(jnp.sum(exp_a, axis=axis, keepdims=True))[0]
    log_sum = safe_log(exp_a_sum) + a_max
    if not keepdims:
        log_sum = log_sum.squeeze(axis=axis)
    return log_sum

from netket.utils.mpi import n_nodes, MPI, MPI_py_comm, MPI_jax_comm

def mpi_max_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise logical OR of an array or a scalar across all MPI
    processes, effectively equivalent to an elementwise any

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.MAX, comm=comm, token=token)
    

def mpi_mean_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise mean of an array or a scalar across all MPI processes
    of a jax array.

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    res, token = mpi_sum_jax(x, token=token, comm=comm)
    return res / n_nodes, token


def mpi_sum_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm, token=token)