from typing import Callable, Tuple, Any
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple
from netket.utils.types import DType, PyTree, Array, Callable
from functools import partial
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import itertools

from functools import partial

from src.utils import get_distance_matrix, array_init, remove_diag
from src.quantumdot.matrix_elements_lg import get_mus, compute_laguerre_orbitals

import netket.nn as nknn
import netket.jax as nkjax

def lecun_normal_scale(std=1.0, **kwargs):
    lecun_normal = jax.nn.initializers.lecun_normal(**kwargs)

    def _lecun_normal_scale(key, shape, dtype):
        out = lecun_normal(key, shape, dtype) * std
        return out.astype(dtype)

    return _lecun_normal_scale


DEFAULT_HIDDEN_ACTIVATION = nknn.gelu 
DEFAULT_OUTPUT_ACTIVATION = nknn.gelu 
DEFAULT_KERNEL_INIT =  lecun_normal_scale(std=1e-1) # not too small for gradients

def make_list(x):
    if hasattr(x, "__len__"):
        return list(x)
    else:
        return [x]


@partial(jax.jit, static_argnames=('n_max',))
def hermite_till_nmax(x, n_max=1):
    # make an extra last axis with the hermitians
    hermites = []
    h0 = jnp.ones_like(x)
    hermites.append(h0)
    if n_max >= 1:
        h1 = 2*x
        hermites.append(h1)
    if n_max >= 2:
        for n in range(2, n_max+1):
            h = 2*x*hermites[-1] - 2*(n-1)*hermites[-2]
            hermites.append(h)
    return jnp.stack(hermites, axis=-1)

def hermite_comb_nd(hermites, ns):
    sdim = len(ns)
    assert hermites.shape[-2] == sdim
    res = 1.
    for d, nd in enumerate(ns):
        res = res*hermites[...,d,nd]
    return res

def get_ns_list(sdim, n_max):
    l = [n for n in itertools.product(range(n_max+1), repeat=sdim) if sum(n)<= n_max]      
    l = list(sorted(l, key=sum))  
    return l

def normal_initializer(stddev=0.01, mean=0.0, dtype=jnp.float64):
    jax_normal = jax.nn.initializers.normal(stddev=stddev, dtype=dtype)
    def _init(*args, **kwargs):
        return jax_normal(*args, **kwargs) + mean
    return _init        

class HOOrbs(nn.Module): # actually HF version of that
    # log orbitals
    n_max: int
    n_per_spin: Tuple[int,int]
    param_dtype: DType = jnp.float64
    Ecutoff: int = None
    hartree_fock: bool = True

    @nn.compact
    def __call__(self, x, ri=None, **kwargs): 
        n_up, n_dn = self.n_per_spin
        n_particles = sum(self.n_per_spin)
        n_particles_max = max(self.n_per_spin)
        assert x.shape[-2] == n_particles
        sdim = x.shape[-1]
        x_scaled = x.reshape(-1, n_particles, sdim)
        if ri is not None: # unscaled version
            x = ri.reshape(-1, n_particles, sdim)

        mus = get_mus(self.n_max, Ecutoff=self.Ecutoff, sdim=sdim, m_constraint=False)
        assert mus.shape[0] >= n_particles_max, \
            f"need higher n_max: {len(mus)}"
        assert mus.shape[-1] == 3
        orbitals = compute_laguerre_orbitals(x_scaled, mus)

        if not self.hartree_fock:
            orbitals = orbitals[...,:n_particles_max]

        log_orbitals = jnp.log(orbitals.astype(complex)) # work in logspace
        n_orbs = log_orbitals.shape[-1]

        if self.hartree_fock:
            mo_coeff_init = jnp.eye(n_orbs, n_particles_max)
            assert mo_coeff_init.shape == (n_orbs, n_particles_max)
            mo_coeff = self.param(
                "mo_coeff", 
                array_init(mo_coeff_init, noise=1e-3), 
                (n_orbs, n_particles_max), 
                self.param_dtype, 
            )
            # make it hartree fock, but do it in LOGSPACE (!!!)
            # vmap over the orbitals to create
            def _make_log_mo_orbs(cs):
                return nkjax.logsumexp_cplx(log_orbitals, b=cs, axis=-1)
            log_mo_orbs = jax.vmap(_make_log_mo_orbs, in_axes=-1, out_axes=-1)(mo_coeff)
        else:
            log_mo_orbs = log_orbitals

        return log_mo_orbs # logspace  

class HOModel(nn.Module):
    orbitals: Callable
    n_per_spin: Tuple
    sdim: int
    param_dtype: DType = float
    jastrow: Callable = None
    orbital_backflow: Callable = None

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-1]
        N = sum(self.n_per_spin)
        x = x.reshape(-1, N, self.sdim)
        si = np.array([+1]*self.n_per_spin[0] + [-1]*self.n_per_spin[1])

        rij_diff, rij_dist = get_distance_matrix(x)
        rij_dist = rij_dist[...,None]
        ri = x
        ri_norm = jnp.linalg.norm(ri, axis=-1, keepdims=True)**2 # only the square is differentiable everywhere

        x_scaled = x.copy()
        x_slater = x_scaled

        orb_bf = self.orbital_backflow(ri, rij_dist, rij_diff=rij_diff, si=si, ri_norm=ri_norm)
        log_psi = LogSlaterDet(n_per_spin=self.n_per_spin, orbitals=self.orbitals, take_exp=True)(x_slater, ri=x, orbital_backflow=orb_bf)

        if self.jastrow is not None:
            jastrow = make_list(self.jastrow)
            assert len(jastrow) in (1, 2)  # complex
            J = 0.            
            for jastrow_fn, c in zip(jastrow, [1, 1j*2*np.pi]):
                J = J + c*jastrow_fn(
                    rij_dist, si=si, rij_diff=rij_diff, ri=ri, ri_norm=ri_norm, x_scaled=x_scaled
                )
            log_psi = log_psi + J

        return log_psi.reshape(*batch_shape)


class BasisJastrow(nn.Module):
    # log orbitals
    n_max: int
    n_per_spin: Tuple[int,int]
    single_sigma: bool = False
    param_dtype: DType = jnp.float64
    complex_coeff: bool = True
    Ecutoff: int = None
    vary_sigma: bool = True
    laguerre: bool = True

    @nn.compact
    def __call__(self, rij_dist, rij_diff=None, ri=None, ri_norm=None, si=None, x_scaled=None):
        n_up, n_dn = self.n_per_spin
        n_particles = sum(self.n_per_spin)
        n_particles_max = max(self.n_per_spin)

        # we only work with ri
        assert ri is not None
        x = ri
        assert x.shape[-2] == n_particles
        sdim = x.shape[-1]
        x = x.reshape(-1, n_particles, sdim)
        x_scaled = x_scaled.reshape(-1, n_particles, sdim)

        mus = get_mus(self.n_max, Ecutoff=self.Ecutoff, sdim=sdim, m_constraint=False)
        assert mus.shape[0] >= n_particles_max, \
            f"need higher n_max: {len(mus)}"
        assert mus.shape[-1] == 3
        basis_single_body = compute_laguerre_orbitals(x_scaled, mus)
        n_basis_single = basis_single_body.shape[-1]

        # now combine all these bases
        basis = jax.vmap(basis_combine_jastrow)(basis_single_body)

        n_basis = basis.shape[-1]
        # sum over particle pairs BEFORE HF
        basis = basis.mean(axis=-2)

        jastrow_coeff = self.param(
            "jastrow_coeff",
            jax.nn.initializers.normal(stddev=1e-5),
            (n_basis,), # just project to a scalar (!)
            self.param_dtype,
        )

        J = basis.dot(jastrow_coeff)
        return J # logspace  


@jax.jit
def basis_combine_jastrow(single_body_basis):  # in LOGSPACE(!)
    assert (
        single_body_basis.ndim == 2
    ), f"should be single sample, got {single_body_basis.shape}, maybe forgot to vmap over the batch size?"
    N, Nb = single_body_basis.shape
    # input = \chi_mu(ri)
    # in the form [i, mu] = [N, Nb]
    # we must combine them to the form
    # output = \chi_mu (ri) \chi_nu (rj)
    # in the form [i < j, mu*nu], for example (only off diag elements to avoid double counting)

    # first select what to combine (i < j)
    idxs = jnp.triu_indices(N, k=1)
    chi_mu_i = single_body_basis[idxs[0], :]
    chi_nu_j = single_body_basis[idxs[1], :]
    # note: now the first axis is the same length (i*j, i<j)

    # now combine the last axis (vmap over all particle combinations)
    chi_munu_ij = jax.vmap(jnp.meshgrid)(chi_mu_i, chi_nu_j)
    # vmap over particles again
    chi_munu_ij = jax.vmap(lambda a, b: jnp.ravel(a * b))(
        *chi_munu_ij
    )  # product over mu and nu, in real space
    # output is now of dimension [i<j, mu*nu] # last axis must later be summed over
    assert chi_munu_ij.shape[-1] == Nb * Nb, f"got {chi_munu_ij.shape} vs Nb={Nb}"
    assert chi_munu_ij.shape[0] == int(
        N * (N - 1) / 2
    ), f"got {chi_munu_ij.shape} vs N={N}"

    return chi_munu_ij


class BasisOrbitalBackflow(nn.Module):
    # log orbitals
    n_orbitals: int
    n_max: int
    n_per_spin: Tuple[int,int]
    param_dtype: DType = jnp.float64
    Ecutoff: int = None

    @nn.compact
    def __call__(self, ri, rij_dist, ri_norm=None, rij_diff=None, si=None):
        n_up, n_dn = self.n_per_spin
        n_particles = sum(self.n_per_spin)
        n_particles_max = max(self.n_per_spin)

        # we only work with ri
        x = ri
        assert x.shape[-2] == n_particles
        sdim = x.shape[-1]
        x = x.reshape(-1, n_particles, sdim)

        mus = get_mus(self.n_max, Ecutoff=self.Ecutoff, sdim=sdim, m_constraint=False)
        # assert mus.shape[0] >= n_particles_max, \
        #     f"need higher n_max: {len(mus)}"
        assert mus.shape[-1] == 3
        basis_single_body = compute_laguerre_orbitals(x, mus)

        # now combine all these bases
        basis = jax.vmap(basis_combine_backflow)(basis_single_body)

        n_basis = basis.shape[-1]
        # sum over OTHER particles before
        basis = remove_diag(basis, has_aux_axis=True) # remove diag from axis 1 and 2 (has mu*nu axis)
        basis = basis.mean(axis=-2) # sum over the other particles

        backflow_coeff = self.param(
            "backflow_coeff",
            jax.nn.initializers.normal(stddev=1e-2),
            (n_basis, self.n_orbitals), # just project to a scalar (!)
            self.param_dtype,
        )

        BF = basis.dot(backflow_coeff)
        assert BF.shape == (*x.shape[:-1], self.n_orbitals), f"backflow got output shape {BF.shape}"
        
        return BF    

@jax.jit
def basis_combine_backflow(single_body_basis): # in LOGSPACE(!)
    # could be made from Jastrow and symmetrizing (!)
    assert single_body_basis.ndim == 2, f"should be single sample, got {single_body_basis.shape}, maybe forgot to vmap over the batch size?"
    N, Nb = single_body_basis.shape
    # input = \chi_mu(ri)
    # in the form [i, mu] = [N, Nb]
    # we must combine them to the form
    # output = \chi_mu (ri) \chi_nu (rj)
    # in the form [i, j, mu*nu], for example (keep i, j axes because we will later sum over j)

    # vmap over i and j and take all combinations of mu and nu
    def meshgrid_and_prod(a, b):
        # input = [mu] [nu] --> [mu*nu] with items being the product of terms
        mg = jnp.stack(jnp.meshgrid(a, b), axis=0)
        return jnp.prod(mg, axis=0).ravel()

    chi_munu_ij = jax.vmap(
        lambda chi_i_mu: jax.vmap(
            lambda chi_j_nu: meshgrid_and_prod(chi_i_mu, chi_j_nu)
        )(single_body_basis)
    )(single_body_basis)

    # output is now of dimension [i, j, mu*nu] # last axis must later be summed over
    assert chi_munu_ij.shape[2] == Nb*Nb, f"got {chi_munu_ij.shape} vs Nb={Nb}"
    assert chi_munu_ij.shape[0] == chi_munu_ij.shape[1] == N, f"got {chi_munu_ij.shape} vs N={N}"

    return chi_munu_ij


class LogSlaterDet(nn.Module):
    n_per_spin: Tuple[int]
    orbitals: Callable
    take_exp: bool = True

    @nn.compact
    def __call__(
        self, x, orbital_backflow=None, **kwargs
    ):  # ARE NOT IN LOGSPACE FOR NOW (!!!)
        """x: (..., n_particles, s_dim)"""
        if not self.orbitals:
            raise ValueError(f"Empty LogSlaterDet module {self.name}.")
        N = sum(self.n_per_spin)
        assert x.shape[-2] == N

        orbs = self.orbitals(x, **kwargs)

        # orbs are in logspace
        if self.take_exp:
            orbs = jnp.exp(orbs)

        if orbital_backflow is not None:
            assert (
                orbs.shape == orbital_backflow.shape
            ), f"got {orbs.shape} vs {orbital_backflow.shape}"
            orbs *= 1 + orbital_backflow  # let's make it a perturbation (!!!)

        mats = jnp.split(orbs, [self.n_per_spin[0]], axis=-2)
        mats = [m[..., :n] for m, n in zip(mats, self.n_per_spin)]
        assert mats[0].shape[-2] == mats[0].shape[-1], f"got {mats[0].shape}"
        assert mats[1].shape[-2] == mats[1].shape[-1], f"got {mats[1].shape}"

        logslaterdet = 0.0
        for mat in mats:
            logslaterdet += nkjax.logdet_cmplx(mat)

        return logslaterdet
