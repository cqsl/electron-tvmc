import tqdm
import numpy as np
import scipy
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy import integrate
from functools import cache, partial
import itertools
import netket.jax as nkjax
import numba as nb
import netket as nk


from scipy.special import comb
import math

from src.quantumdot.laguerre import eval_genlaguerre

# REF: https://iopscience.iop.org/article/10.1088/0953-8984/10/3/013/pdf?casa_token=dDX7p1xN48wAAAAA:IDprRdRr7dCoAUuc5xDL1zK85pVAhmcLO_p6OpPxrQCwzG_wNVz5eUPG8_QDn7hueBM3otjN9qU

nk.config.netket_experimental_fft_autocorrelation = True

@nb.njit
def compute_orbital_energy(mu):
    s1, n1, m1 = mu
    return 1+np.abs(m1)+2*n1

# @nb.njit
def compute_core_mel(mu, nu, omega=1.0, separate=False):
    s1, n1, m1 = mu
    s2, n2, m2 = nu
    if s1 != s2:
        return (0., 0., omega**2) if separate else 0.
    if not separate and np.abs(omega - 1) < 1e-12:
        if n1 != n2:
            return 0.
        return compute_orbital_energy(nu)
    else:
        omega = float(omega)
        t1 = compute_kinetic_mel(mu, nu)
        t2 = 0.5*_compute_monopole_mel(mu, nu)
        if separate:
            return t1, t2, omega**2
        else:
            return t1 + (omega**2)*t2


def get_mus(n_max, Ecutoff=np.inf, m_max=None, n_min=0, sdim=2, polarized=True, m_constraint=False):
    assert polarized
    if m_max is None:
        m_max = 2*n_max # max value for basis function
        # m_max = n_max # max value for basis function
    ns = np.arange(n_max+1)
    ns = ns[ns>=n_min]
    ms = np.arange(-m_max, +m_max+1)
    mus = np.meshgrid(ns, ms)
    mus = list(map(lambda x: x.T.ravel(), mus))
    mus = np.stack(mus, axis=-1)
    
    if m_constraint:
        # THIS CONSTRAINT ACTUALLY DOES NOT HOLD, OTHERWISE DO NOT REPRODUCE NON-INTERACTING!!!
        mus = mus[np.abs(mus[:,1]) <= mus[:,0]+1e-3] # |m| <= n (basis function constraint)
    
    mus = np.pad(mus, [[0, 0], [1, 0]], constant_values=-1) # add some spin
    
    energies = np.array(list(map(compute_orbital_energy, mus)))
    idxs = np.argsort(energies)
    energies = energies[idxs]
    mus = np.array(mus)[idxs,:]

    mus = mus[energies <= Ecutoff+1e-3,:]

    return mus


@nb.njit
def _gamma(j1, j4, m1, m4, first):
    if first:
        sign = +1
    else:
        sign = -1
    return int(np.rint(j1 + j4 + (np.abs(m1)+sign*m1)/2 + (np.abs(m4)-sign*m4)/2))

def __comb_exact(N, k):
    return scipy.special.comb(N, k, exact=False)

@nb.njit
def _comb(N, k):
    N = int(N)
    k = int(k)
    if k > N or N < 0 or k < 0:
        return 0
    M = N + 1
    nterms = min(k, N - k)
    numerator = 1
    denominator = 1
    for j in range(1, nterms + 1):
        numerator *= M - j
        denominator *= j
    return numerator // denominator

assert np.isclose(__comb_exact(6, 3), _comb(6, 3))

def __gamma_function_exact(z):
    return scipy.special.gamma(z)

@nb.njit
def _gamma_function(x):
    # DONT USE THE PART BELOW, IT GIVES WEIRD RESULTS FOR INTEGERS!!!
    # if x >=0 and x <= 20 and np.abs(np.rint(x) - x) < 1e-12:
    #     print("Got x = ", x, int(x), int(x)-1, factorial(int(x) - 1))
    #     return factorial(int(x) - 1)
    # else:
    #     print("Using math:", x)
    return math.gamma(x)

LOOKUP_TABLE = np.array([
1, 1, 2, 6, 24, 120, 720, 5040, 40320,
362880, 3628800, 39916800, 479001600,
6227020800, 87178291200, 1307674368000,
20922789888000, 355687428096000, 6402373705728000,
121645100408832000, 2432902008176640000], dtype='int64')

@nb.njit
def factorial(n):
    if n > 20:
        return _gamma_function(n+1)
    else:
        return LOOKUP_TABLE[n]

assert np.isclose(factorial(5), scipy.special.factorial(5))



assert np.isclose(_gamma_function(3.5), __gamma_function_exact(3.5))

@nb.njit
def _sum_generator_4(max_vals, inclusive=True):
    add = int(inclusive)
    for a in range(max_vals[0]+add):
        for b in range(max_vals[1]+add):
            for c in range(max_vals[2]+add):
                for d in range(max_vals[3]+add):
                    yield a, b, c, d

@nb.njit
def _sum_generator_2(max_vals, inclusive=True):
    add = int(inclusive)
    for a in range(max_vals[0]+add):
        for b in range(max_vals[1]+add):
            yield a, b

                    
# @nb.njit
# def _sum_generator(max_vals, inclusive=True):
#     assert len(max_vals) == 4
#     add = int(inclusive)
#     out = []
#     for a in range(max_vals[0]+add):
#         for b in range(max_vals[1]+add):
#             for c in range(max_vals[2]+add):
#                 for d in range(max_vals[3]+add):
#                     out.append([a, b, c, d])
                    
#     out_arr = np.zeros((len(out), 4), dtype=np.int64)
#     for i in range(out_arr.shape[0]):
#         for j in range(out_arr.shape[1]):
#             out_arr[i, j] = out[i][j]
#     return out_arr

@nb.njit
def sign_power(n):
    even = int(n) % 2 == 0
    return +1 if even else -1

@nb.njit
def _compute_coulomb_mel(a, b, c, d):
    s1, n1, m1 = a
    s2, n2, m2 = b
    s3, n3, m3 = c
    s4, n4, m4 = d
    # check spins
    if s1 != s4:
        return 0.
    if s2 != s3:
        return 0.
    # check z projection
    if m1+m2 != m3+m4:
        return 0.
    # first term
    f1 = 1.0
    for _, ni, mi in [a, b, c, d]:
        f1 *= factorial(ni)/factorial(np.abs(mi)+ni)
    f1 = np.sqrt(f1)
    # other terms
    j_sum = 0.
    for j1, j2, j3, j4 in _sum_generator_4([n1, n2, n3, n4]):
        f2 = 1.0
        for j in [j1, j2, j3, j4]:
#             f2 *= (-1)**j / factorial(j)
            f2 *= sign_power(j) / factorial(j)
        f3 = 1.0
        for (_, ni, mi), ji in zip([a, b, c, d], [j1, j2, j3, j4]):
            f3 *= _comb(ni+np.abs(mi), ni-ji)
        g1 = _gamma(j1, j4, m1, m4, True)
        g4 = _gamma(j1, j4, m1, m4, False)
        g2 = _gamma(j2, j3, m2, m3, True)
        g3 = _gamma(j2, j3, m2, m3, False)
        G = g1+g2+g3+g4
        f4 = 1/2**((G+1)/2)
        l_sum = 0.
        for l1, l2, l3, l4 in _sum_generator_4([g1, g2, g3, g4]):
            if (l1+l2) != (l3+l4):
                continue
#             f5 = (-1)**(g2+g3-l2-l3) # why only those?
            f5 = sign_power(g2+g3-l2-l3)
            f6 = 1.0 # already covered
            f7 = 1.0
            for gi, li in zip([g1, g2, g3, g4], [l1, l2, l3, l4]):
                f7 *= _comb(gi, li)
            Lambda = l1+l2+l3+l4
            f8 = _gamma_function(1+Lambda/2)
            f9 = _gamma_function((G-Lambda+1)/2)
            l_sum += f5*f6*f7*f8*f9
        j_sum += f2*f3*f4*l_sum
    return f1*j_sum
    

# @nb.njit
def compute_coulomb_mel(a, b, c, d):
    a = np.array(a, dtype=np.int64)
    b = np.array(b, dtype=np.int64)
    c = np.array(c, dtype=np.int64)
    d = np.array(d, dtype=np.int64)
    return _compute_coulomb_mel(a, b, c, d)



@nb.njit
def _compute_monopole_mel(a, b):
    s1, n1, m1 = a
    s2, n2, m2 = b
    # check spins
    if s1 != s2:
        return 0.
    # check z projection
    if m1 != m2:
        return 0.
    f1 = 0.5
    for _, ni, mi in [a, b]:
        f1 *= factorial(ni)/factorial(np.abs(mi)+ni)
    f1 = np.sqrt(f1)
    #
    j_sum = 0.
    for j1, j2 in _sum_generator_2([n1, n2]):
        f2 = 1.0
        for j in [j1, j2]:
            f2 *= (-1)**j / factorial(j)
        f3 = 1.0
        for (_, ni, mi), ji in zip([a, b], [j1, j2]):
            f3 *= _comb(ni+np.abs(mi), ni-ji)
        g = (np.abs(m1)+np.abs(m2))/2 + 1 + j1 + j2
        f4 = _gamma_function(1+g)
        j_sum += f2*f3*f4
    return f1*j_sum/np.sqrt(2)*2

# @nb.njit
def compute_monopole_mel(a, b):
    a = np.array(a, dtype=np.int64)
    b = np.array(b, dtype=np.int64)
    return _compute_monopole_mel(a, b)
    
@nb.njit
def enumeration_product_nb_2(array):
    for a in enumerate(array):
        for b in enumerate(array):
            yield a, b
@nb.njit    
def enumeration_product_nb_4(array):
    for a in enumerate(array):
        for b in enumerate(array):
            for c in enumerate(array):
                for d in enumerate(array):
                    yield a, b, c, d

# PARALLEL = True

# @nb.njit(parallel=PARALLEL)
def get_core_matrix(basis_functions, omega=1, separate=False):
    n = len(basis_functions)
    matrix = np.zeros((n, n), dtype=np.complex128)
    if separate:
        matrix2 = np.zeros((n, n), dtype=np.complex128)
        for (i, a), (j, b) in itertools.product(enumerate(basis_functions), repeat=2):
            t1, t2, _ = compute_core_mel(a, b, omega=omega, separate=True)    
            matrix[i,j] = t1
            matrix2[i,j] = t2
        return matrix, matrix2, omega**2
    else:
        print("EXACT ROUTE")
        # for (i, a), (j, b) in itertools.product(enumerate(basis_functions), repeat=2):
        for i, a in enumerate(basis_functions):
    #     for (i, a), (j, b) in enumeration_product_nb_2(basis_functions):
            # matrix[i,j] = compute_core_mel(a, b, omega=omega)
            matrix[i,i] = compute_orbital_energy(a)
        return matrix

# @nb.njit
# def _get_coulomb_values(qns):
#     mels = []
#     for a, b, c, d in qns:
#         mels.append(_compute_coulomb_mel(a, b, c, d))
#     return np.array(mels)

# @nb.njit #(parallel=PARALLEL)
def get_coulomb_matrix(basis_functions, use_precomputed=True, folder="tmp_mels"):
    import sys, os
    import pandas as pd
    n = len(basis_functions)
    matrix = np.zeros((n, n, n, n), dtype=np.complex128)
    n_elems = n**4

    print("Compute n_elems = ", n_elems)
    if use_precomputed:
        # load all the relevant data
        n_max = np.max(basis_functions[:,1])
        m_max = np.max(np.abs(basis_functions[:,2]))
        Ecutoff = np.array(list(map(compute_orbital_energy, basis_functions)))
        Ecutoff = np.max(Ecutoff)

        def get_m_max(args):
            # s not included here
            _, ma, _, mb, _, mc, _, md = args
            return max(abs(ma), abs(mb), abs(mc), abs(md))
        
        def get_cutoff(args):
            mua = np.array((-1,)+args[0:2])
            mub = np.array((-1,)+args[2:4])
            muc = np.array((-1,)+args[4:6])
            mud = np.array((-1,)+args[6:8])
            return np.max(list(map(compute_orbital_energy, [mua, mub, muc, mud])))

        def load_df(n):
            fname = f"uuuu_n{n}.parquet"
            fname = os.path.join(folder, fname)
            if os.path.exists(fname):
                print("Loading:", fname)
                df = pd.read_parquet(fname)
                df["m_max"] = df.index.map(get_m_max)
                df["Ecutoff"] = df.index.map(get_cutoff)
                df = df.loc[df["m_max"] <= m_max]
                df = df.loc[df["Ecutoff"] <= (Ecutoff+1e-5)]
                return df
            else:
                print("(!!!!) Missing n_max file:", n)
                return None
        
        data = {n:load_df(n) for n in range(n_max+1)}

        for (i, a), (j, b), (k, c), (l, d) in tqdm.tqdm(itertools.product(enumerate(basis_functions), repeat=4), total=n_elems):
            n = max(a[1], b[1], c[1], d[1])
            df = data[n]
            if df is not None:
                try:
                    mel = df.loc[(*a[1:], *b[1:], *c[1:], *d[1:]), "values"]
                except Exception as e:
                    print("keyerror = ", e)
                    mel = compute_coulomb_mel(a, b, c, d)
            else:
                mel = compute_coulomb_mel(a, b, c, d)
            matrix[i,j,k,l] = mel
    else:
        for (i, a), (j, b), (k, c), (l, d) in tqdm.tqdm(itertools.product(enumerate(basis_functions), repeat=4), total=n_elems):
    #     for (i, a), (j, b), (k, c), (l, d) in enumeration_product_nb_4(basis_functions):
            if np.abs(matrix[i,j,k,l]) < 1e-12:
                mel = np.complex128(compute_coulomb_mel(a, b, c, d))
                matrix[i,j,k,l] = mel
                matrix[l,k,j,i] = mel.conj()
                matrix[j,i,l,k] = mel
                matrix[k,l,i,j] = mel.conj()
    return matrix

# @nb.njit(parallel=PARALLEL)
def get_monopole_matrix(basis_functions):
    n = len(basis_functions)
    matrix = np.zeros((n, n), dtype=np.float64)
    for (i, a), (j, b) in itertools.product(enumerate(basis_functions), repeat=2):
#     for (i, a), (j, b) in enumeration_product_nb_2(basis_functions):
        matrix[i,j] = compute_monopole_mel(a, b)
    return matrix

# def precompute_coulomb_matrix(n_max, m_max=None, n_min=0, folder='tmp_mels'):
#     import pathlib
#     import os, sys
#     import pandas as pd
#     import time
    
#     basis_functions = get_mus(n_max, m_max=m_max).astype(np.int64)
#     idx = np.lexsort([x for x in basis_functions.T[::-1]])
#     basis_functions = basis_functions[idx,:]
#     print("BASIS FUNCTIONS:")
#     print(basis_functions)
#     # need to sort !!! (already done if using get_mus)
#     n = len(basis_functions)
#     n_elems = n**4
    
#     def _to_file(na, nb):
#         pathlib.Path(folder).mkdir(exist_ok=True)
#         df = pd.DataFrame(data)
#         df = df.set_index(["na", "ma", "nb", "mb", "nc", "mc", "nd", "md"])
#         fname = "uuuu_" + str(na) + "_" + str(nb)
#         fname = os.path.join(folder, f"{fname}.parquet")
#         # read what we already have
#         #display(df)
#         if os.path.exists(fname):
#             old_df = pd.read_parquet(fname)
# #             if old_df.shape[0] > 0:
# #                 display(old_df)
# #                 print("!"*100, "already had stuff!!!")
#             df = pd.concat([old_df, df], axis=0)
#             df = df[~df.index.duplicated(keep='first')]
#             df = df.sort_index()
#         print("dumping to fname:", fname, "size = ", df.shape)
#         #display(df)
#         df.to_parquet(fname)
    
#     print("Precompute n_elems = ", n_elems)
#     idx = (0, 0)
#     data = {**{f"n{k}":[] for k in "abcd"}, **{f"m{k}":[] for k in "abcd"}, "values":[]}
#     for (i, a), (j, b) in tqdm.tqdm(itertools.product(enumerate(basis_functions), repeat=2), total=n**2):
#         a, b = tuple(a), tuple(b)
#         na = a[1]
#         nb = b[1]
#         cur_idx = (na, nb)

#         if cur_idx != idx:
#             # save previous df
#             _to_file(*idx)
#             data = {**{f"n{k}":[] for k in "abcd"}, **{f"m{k}":[] for k in "abcd"}, "values":[]}
#             idx = cur_idx
        
#         for (k, c), (l, d) in itertools.product(enumerate(basis_functions), repeat=2):
#             nc = c[1]
#             nd = d[1]
#             if np.all(np.array([na, nb, nc, nd]) < n_min):
#                 continue                
#             c, d = tuple(c), tuple(d)
#             mel = compute_coulomb_mel(a, b, c, d)
#             for mu_idx, mu_val in zip("abcd", [a, b, c, d]):
#                 data[f"n{mu_idx}"].append(mu_val[1])
#                 data[f"m{mu_idx}"].append(mu_val[2])
#             data["values"].append(mel)

#     _to_file(*idx)
            
#     print("all done...")


def precompute_coulomb_matrix(n_max, m_max=None, n_min=0, folder='tmp_mels'):
    import pathlib
    import os, sys
    import pandas as pd
    import time
    
    basis_functions = get_mus(n_max, m_max=m_max).astype(np.int64)
    idx = np.lexsort([x for x in basis_functions.T[::-1]])
    basis_functions = basis_functions[idx,:]
    print("BASIS FUNCTIONS:")
    print(basis_functions)
    # need to sort !!! (already done if using get_mus)
    n = len(basis_functions)
    n_elems = n**4
    
    def _to_file(na, nb):
        pathlib.Path(folder).mkdir(exist_ok=True)
        df = pd.DataFrame(data)
        df = df.set_index(["na", "ma", "nb", "mb", "nc", "mc", "nd", "md"])
        fname = "uuuu_" + str(na) + "_" + str(nb)
        fname = os.path.join(folder, f"{fname}.parquet")
        # read what we already have
        #display(df)
        if os.path.exists(fname):
            old_df = pd.read_parquet(fname)
#             if old_df.shape[0] > 0:
#                 display(old_df)
#                 print("!"*100, "already had stuff!!!")
            df = pd.concat([old_df, df], axis=0)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
        print("dumping to fname:", fname, "size = ", df.shape)
        #display(df)
        df.to_parquet(fname)
    
    print("Precompute n_elems = ", n_elems)
    idx = (0, 0)
    data = {**{f"n{k}":[] for k in "abcd"}, **{f"m{k}":[] for k in "abcd"}, "values":[]}
    for (i, a), (j, b) in tqdm.tqdm(itertools.product(enumerate(basis_functions), repeat=2), total=n**2):
        a, b = tuple(a), tuple(b)
        na = a[1]
        nb = b[1]
        cur_idx = (na, nb)

        if cur_idx != idx:
            # save previous df
            _to_file(*idx)
            data = {**{f"n{k}":[] for k in "abcd"}, **{f"m{k}":[] for k in "abcd"}, "values":[]}
            idx = cur_idx
        
        for (k, c), (l, d) in itertools.product(enumerate(basis_functions), repeat=2):
            nc = c[1]
            nd = d[1]
            if np.all(np.array([na, nb, nc, nd]) < n_min):
                continue                
            c, d = tuple(c), tuple(d)
            mel = compute_coulomb_mel(a, b, c, d)
            for mu_idx, mu_val in zip("abcd", [a, b, c, d]):
                data[f"n{mu_idx}"].append(mu_val[1])
                data[f"m{mu_idx}"].append(mu_val[2])
            data["values"].append(mel)

    _to_file(*idx)
            
    print("all done...")

@jax.jit
def factorial_jax(n):
    return jnp.exp(jax.scipy.special.gammaln(n+1))

@partial(jax.jit, static_argnames=('beta',))
def _laguerre(x, alpha, beta):
    # alpha can also be an array !
    # L^alpha_beta
    if beta < 0:
        raise ValueError()
    lgs = []
    if beta >= 0:
        lgs.append(jnp.ones_like(x))
    if beta >= 1:
        lgs.append(1+alpha-x)
    if beta >= 2:
        for k in range(1, beta): # provides k+1 laguerre
            lgs.append(
                ((2*k+1+alpha-x)*lgs[-1] - (k+alpha)*lgs[-2])/(k+1)
            )
    return lgs[-1]
    

@partial(jax.jit, static_argnames=('beta',))
def laguerre_jax(x, alpha, beta):
    return _laguerre(x, alpha, beta)
    
@partial(jax.jit, static_argnames=('n',))
def compute_orbital(x, n, m):
    # must have sdim
    r = jnp.linalg.norm(x, axis=-1)
    is_zero = jnp.abs(r) < 1e-14
    r_valid = jnp.where(is_zero, 1.0, r)
    r2 = r**2
    x_normed = jnp.where(is_zero[...,None], x, x/r_valid[...,None])
    phi = jnp.arctan2(x_normed[...,1], x_normed[...,0])
    lag = laguerre_jax(r2, jnp.abs(m), n)
    prefac = jnp.sqrt(1/(2*np.pi)*2*scipy.special.factorial(n)/factorial_jax(jnp.abs(m)+n))
    return prefac*jnp.power(r, jnp.abs(m))*jnp.exp(1j*m*phi-r2/2)*lag


def compute_laguerre_orbitals(x, mus):
    assert mus.shape[-1] == 3
    # mus = [s, n, m]
    # ss = mus[:,0]
    ns = mus[:,1]
    ms = mus[:,2]
    nsfac = np.array(list(map(scipy.special.factorial, ns)))
    mnsfac = np.array(list(map(scipy.special.factorial, np.abs(ms)+ns)))

    @jax.jit
    def to_polar(x):
        r = jnp.linalg.norm(x, axis=-1)
        is_zero = jnp.abs(r) < 1e-14
        r_valid = jnp.where(is_zero, 1.0, r)
        r2 = r**2
        x_normed = jnp.where(is_zero[...,None], x, x/r_valid[...,None])
        phi = jnp.arctan2(x_normed[...,1], x_normed[...,0])
        return r, r2, phi
    
    r, r2, phi = to_polar(x)

    # must have sdim
    @jax.jit
    def compute_coeffs(m, nfac, mnfac):
        prefac = jnp.sqrt(1/(2*np.pi)*2*nfac/mnfac)
        return prefac*jnp.power(r, jnp.abs(m))*jnp.exp(1j*m*phi-r2/2)
    
    prefacs = jax.vmap(compute_coeffs, out_axes=-1)(ms, nsfac, mnsfac)
    lags = eval_genlaguerre(ns, np.abs(ms), r2)

    assert prefacs.shape == lags.shape, f"got {prefacs.shape} vs {lags.shape}"
    orbs = prefacs*lags
    return orbs
    # orbs = []
    # for n, m in zip(ns, ms):
    #     orbs.append(
    #         compute_orbital(x, n, m)
    #     )
    # return jnp.stack(orbs, axis=-1)

def jacrev(f):
    def jacfun(x):
        y, vjp_fun = nkjax.vjp(f, x)
        if y.size == 1:
            eye = jnp.eye(y.size, dtype=x.dtype)[0]
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        else:
            eye = jnp.eye(y.size, dtype=x.dtype)
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        return J

    return jacfun


def jacfwd(f):
    def jacfun(x):
        jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
        eye = jnp.eye(len(x), dtype=x.dtype)
        J = jax.vmap(jvp_fun, in_axes=0)(eye)
        return J

    return jacfun

def wrap_fn_scipy(fn):
    def _wrapped_scipy_fn(*args):
        x = jnp.stack(args, axis=-1)
        return fn(x)
    return _wrapped_scipy_fn

def laplacian(f):
    df_dx = jacrev(f)
    @jax.jit
    def _laplacian(x):
        if x.ndim > 1:
            print("WARNING: VMAPPING LAPLACIAN!!!", x.shape)
            return jax.vmap(_laplacian)(x)
        assert x.ndim == 1, f"got x = {x.shape}"
        df_dx2 = jnp.diag(jacfwd(df_dx)(x)[0].reshape(x.shape[0], x.shape[0]))
        return jnp.sum(df_dx2)
    return _laplacian


def integrate_1d(fn, **kwargs):
    return integrate_nd(fn, 1, **kwargs)
def integrate_2d(fn, **kwargs):
    return integrate_nd(fn, 2, **kwargs)
def integrate_4d(fn, **kwargs):
    return integrate_nd(fn, 4, **kwargs)

def integrate_nd(fn, n, radius=5, key=None, n_samples=1024*64, chunk_size=1024, return_err=True):
    if key is None:
        key = jax.random.PRNGKey(0)
    x = radius*jax.random.ball(key, n, p=2, shape=(n_samples,))
    volume = (np.pi)**(n/2) / scipy.special.gamma(n/2+1) * radius**n
    values = nkjax.vmap_chunked(fn, chunk_size=chunk_size)(x)*volume
    mean = jnp.mean(values)
    err = jnp.std(values) / np.sqrt(values.shape[0])
    if return_err:
        return mean, err
    else:
        return mean

def compute_density_matrix(n_electron, c, polarized=True):
    r"""Compute the density matrix (currently for fully polarized case."""
    if polarized:
        select_max = n_electron
        factor = 1
    else:
        assert False
        assert n_electron % 2 == 0
        select_max = n_electron // 2
        factor = 2
    p = factor*jnp.dot(c[:, :select_max], c[:, :select_max].conj().T) # this one makes most sense and is pyscf and works with .conj() fock matrix
    # p = factor*jnp.dot(c[:, :select_max].conj(), c[:, :select_max].T) # this one works with the state vector
    return p

# good HF summary: https://www.tandfonline.com/doi/epdf/10.1080/00268970701757875?needAccess=true&role=button

def compute_veff(rep_tensor, dm, kappa=1.0):
    # convention: v_eff contains kappa (!!!)
    dm = dm.conj() # from definition (!!!!!!!!!!!)
    j = jnp.einsum("pqrs,qr->ps", rep_tensor, dm)
    k = jnp.einsum("pqrs,qr->ps", jnp.swapaxes(rep_tensor, -2, -1), dm)
    # j same spin + j opp spin  - k 
    # reduced here to j same spin - k
    v_eff = j - k # 0.5 from taking i \neq j instead of i < j
    return v_eff*kappa
    
def get_fock_matrix(h_core, v_eff):
    # usual reasoning: There is a double contribution from the coulomb operator 
    # as we are considering closed shell systems where 
    # there are two electrons per orbital but the exhange operator is only 
    # considered once as this interaction can only be between electrons with like-spins
    fock_matrix = h_core + v_eff
    return fock_matrix

def compute_tensors(basis_functions, use_precomputed=False, omega=1, separate=False):
    h_core = get_core_matrix(basis_functions, omega=omega, separate=separate)
    rep_tensor = get_coulomb_matrix(basis_functions, use_precomputed=use_precomputed)
    return h_core, rep_tensor


def scf(basis_functions, n_electron, kappa=1.0, n_steps=1000, tol=1e-12, h_core=None, rep_tensor=None, verbose=False):
    # for orthonormal orbitals !!!
    n = len(basis_functions)
    if h_core is None:
        h_core = get_core_matrix(basis_functions)
    if rep_tensor is None:
        rep_tensor = get_coulomb_matrix(basis_functions)

#     eigvals, w_fock = np.linalg.eigh(
#         h_core
#     )  # initial guess for the scf problem (should be simply a diagonal matrix here !)
    eigvals = h_core
    w_fock = np.eye(n)
    coeffs = w_fock
    
    p = compute_density_matrix(n_electron, coeffs)

    for istep in tqdm.tqdm(range(n_steps)):
        v_eff = compute_veff(rep_tensor, p, kappa=kappa)
        fock_matrix = get_fock_matrix(h_core, v_eff)
        eigvals, w_fock = jnp.linalg.eigh(fock_matrix)
        coeffs = w_fock

        p_update = compute_density_matrix(n_electron, coeffs)

        if verbose:
            E = get_hf_energy(h_core, rep_tensor, p, kappa=kappa)
            print("Energy = ", E)

        if tol is not None and np.linalg.norm(p_update - p) <= tol:
            print(f"Converged in {istep} steps.")
            break

        p = p_update

    return eigvals, coeffs, fock_matrix, h_core, rep_tensor, v_eff

# scf(mus, 2)


@nb.njit
def compute_kinetic_mel(mu, nu):
    s1, n1, m1 = mu
    s2, n2, m2 = nu
    if s1 != s2:
        return 0.
    if m1 != m2:
        return 0.
    m = m1
    # first term
    f1 = 1.0
    for _, ni, mi in [mu, nu]:
        f1 *= factorial(ni)/factorial(np.abs(mi)+ni)
    f1 = np.sqrt(f1)
    # rest
    j_sum = 0.
    for j1, j2 in _sum_generator_2([n1, n2]):
        if j1 == 0 and j2 == 0 and m == 0:
            j_sum += 1.0 # Limit[(m^2 + Abs[m]) Gamma[Abs[m]], m -> 0] = 1.0
        else:
            f5 = m**2 + np.abs(m) - (j1**2 + j2**2) + (j1 + j2) + 2*j1*j2
    #         if f5 == 0:
    #             continue
            f2 = 1.0
            for j in [j1, j2]:
                f2 *= sign_power(j) / factorial(j)
            f3 = 1.0
            for (_, ni, mi), ji in zip([mu, nu], [j1, j2]):
                f3 *= _comb(ni+np.abs(mi), ni-ji)
            g = np.abs(m) + j1 + j2
            f4 = _gamma_function(g)
            # print(f1, f2, f3, f4, f5)
            j_sum += f2*f3*f4*f5
            # print(j_sum)
    return 0.5*f1*j_sum

def obs_from_dm(obs_mat, P):
    assert obs_mat.shape == P.shape
    assert obs_mat.ndim == P.ndim == 2
    # return jnp.einsum("pq,pq", obs_mat, P) # Tr[O\rho]
    # return jnp.einsum("pq,qp", obs_mat, P) # Tr[O\rho]
    return jnp.einsum("pq,qp", P, obs_mat) # Tr[O\rho]

def get_hf_energy(h_core, rep_tensor, dm, kappa=1.0):
    r"""Compute the total single-Slater Hartree-Fock energy."""
    v_eff = compute_veff(rep_tensor, dm, kappa=kappa)
    H = h_core + 0.5*v_eff
    energy = obs_from_dm(H, dm)
    return energy.reshape(())