from typing import Callable, Tuple, Any
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple
from flax import struct
from netket.utils.types import DType, PyTree, Array, Callable
import copy
from functools import partial
import numpy as np
import netket as nk
import netket.experimental as nkx
import jax
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import itertools

from functools import partial
import json
import optax

from src.utils import get_distance_matrix
from src.model import _mat_to_logdet
from src.nn import make_complex_params, array_init
from src.hamiltonian import get_el_el_potential_energy 
from src.quantumdot.slater import LogSlaterDet
from src.utils import get_distance_matrix
from src.utils import get_spin_spin_matrix, remove_diag, pick_triu
from src.nn import real_to_complex, make_complex_params, array_init, ssp, lecun_normal_scale, poly_logcosh
from src.cusp import el_el_cusp, el_ion_cusp
from src.features import get_tanh_feature
from src.model import psiformer_scale, Attention, ProjectiveMapping
from src.quantumdot.matrix_elements_lg import get_mus, compute_laguerre_orbitals

from src.utils import print_mpi, complex_dtype
import netket.nn as nknn
import netket.jax as nkjax

DEFAULT_HIDDEN_ACTIVATION = nknn.gelu #lambda x: x #nknn.silu #nknn.gelu
DEFAULT_OUTPUT_ACTIVATION = nknn.gelu #nknn.silu
DEFAULT_KERNEL_INIT =  lecun_normal_scale(std=1e-1) # not too small for gradients

print_mpi("DEFAULTS:", DEFAULT_HIDDEN_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION, DEFAULT_KERNEL_INIT)

def logsumexp_cplx(a, b=None, **kwargs):
    a = a.astype(complex)
    a, sgn = jax.scipy.special.logsumexp(a, b=b, **kwargs, return_sign=True)
    a = a + jnp.where(sgn < 0, 1j * jnp.pi, 0j)
    return a

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


# @partial(jax.jit, static_argnames=('full_det','n_per_spin'))
# def _make_slater_det(mo_orbs, n_per_spin, full_det=False):
#     n_up, n_dn = n_per_spin
#     n_particles_max = max(n_per_spin)

#     assert mo_orbs.shape[-1] == n_particles_max
#     mo_matrix_up = mo_orbs[..., :n_up, :n_up] # last axis is in case there are too many
#     mo_matrix_dn = mo_orbs[..., n_up:, :n_dn]
#     # print("mo_matrices:", mo_matrix_up.shape, mo_matrix_dn.shape)
#     if full_det:
#         batch_shape = mo_matrix_up.shape[:-2]
#         mo_matrix_up = jnp.concatenate(
#             [mo_matrix_up, jnp.zeros(batch_shape + (n_up, n_dn))], axis=-1)
#         mo_matrix_dn = jnp.concatenate(
#             [jnp.zeros(batch_shape + (n_dn, n_up)), mo_matrix_dn], axis=-1)
#         mat = jnp.concatenate([mo_matrix_up, mo_matrix_dn], axis=-2)
#         log_slater = _mat_to_logdet(mat)
#     else:
#         log_slater = _mat_to_logdet(mo_matrix_up) + \
#             _mat_to_logdet(mo_matrix_dn)
#     return log_slater


class FlexOrbs(nn.Module):
    # logspace
    n_per_spin: Tuple[int,int]
    single_sigma: bool = False
    vary_sigma: bool = True
    param_dtype: DType = jnp.float64
    complex_coeff: bool = False
    hidden_dim: int = 8
    use_mlp: bool = True
    
    @nn.compact
    def __call__(self, x, r=None, **kwargs):
        n_up, n_dn = self.n_per_spin
        n_particles = sum(self.n_per_spin)
        n_particles_max = max(self.n_per_spin)
        assert x.shape[-2] == n_particles
        sdim = x.shape[-1]
        x = x.reshape(-1, n_particles, sdim)
        if r is None:
            r = jnp.linalg.norm(x, axis=-1)
        else:
            assert r.shape == x.shape[:-1], f"got {r.shape} vs {x.shape}"
        
        n_orbs = n_particles_max
        assert n_orbs >= n_particles_max
        # create a higher
        # orbitals = [nk.models.MLP(param_dtype=self.param_dtype, hidden_dims=(self.hidden_dim,), name=f"MLPorb{iorb}")(x) for iorb in range(n_orbs)]
        # orbitals = jax.vmap(lambda i: jax.lax.switch(i, orbitals, x), out_axes=-1)(np.arange(n_orbs)) # does things in parallel?
        orbitals = [
            ProjectiveMapping(param_dtype=self.param_dtype, complex_coeff=self.complex_coeff, 
                              embedding_dim=self.hidden_dim, hidden_dims=None, 
                              use_mlp=self.use_mlp, 
                              add_bias=False, 
                              name=f"MLPorb{iorb}")(x)
            for iorb in range(n_orbs)
        ]
        orbitals = jnp.stack(orbitals, axis=-1)
        
        log_orbitals = jnp.log(orbitals.astype(complex)) # work in logspace
        n_orbs = log_orbitals.shape[-1]
        if self.vary_sigma:
            sigma = self.param(
                "exp_sigma", 
                normal_initializer(mean=1.0, stddev=1e-3), 
                () if self.single_sigma else (n_orbs,), 
                self.param_dtype
            )
        else:
            sigma = jnp.ones(() if self.single_sigma else (n_orbs,))
        # sigma = sigma*0 + 1.0
        # add bc dimension for sdim
        if self.single_sigma:
            sigma = sigma[...,None]
        exp_arg = (sigma*r[...,None])**2 # bc log_orbitals
        log_orbitals += -0.5*exp_arg # logspace

        log_mo_orbs = log_orbitals

        return log_mo_orbs # logspace  
        

class HOOrbs(nn.Module): # actually HF version of that
    # log orbitals
    n_max: int
    n_per_spin: Tuple[int,int]
    single_sigma: bool = False
    vary_sigma: bool = True
    param_dtype: DType = jnp.float64
    complex_coeff: bool = True
    Ecutoff: int = None
    hartree_fock: bool = True
    laguerre: bool = True


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
        
        if self.laguerre:
            mus = get_mus(self.n_max, Ecutoff=self.Ecutoff, sdim=sdim, m_constraint=False)
            assert mus.shape[0] >= n_particles_max, \
                f"need higher n_max: {len(mus)}"
            assert mus.shape[-1] == 3
            orbitals = compute_laguerre_orbitals(x_scaled, mus)
        else:
            ns = np.array(get_ns_list(sdim, self.n_max))
            if self.Ecutoff is not None:
                # print("Cutting off from", ns.shape)
                Es = ns.sum(axis=-1) + sdim/2
                idx_cutoff = Es <= self.Ecutoff
                ns = ns[idx_cutoff,:]
                # print("To", ns.shape, "with max n = ", np.max(ns))
            assert len(ns) >= n_particles_max, \
                f"need higher n_max: {len(ns)}"
            max_n_in_basis = np.max(ns)
            hermites = hermite_till_nmax(x_scaled, n_max=max_n_in_basis)
            orbitals = jax.vmap(hermite_comb_nd, in_axes=(None, 0), out_axes=-1)(hermites, ns)

        if not self.hartree_fock:
            orbitals = orbitals[...,:n_particles_max]

        log_orbitals = jnp.log(orbitals.astype(complex)) # work in logspace
        n_orbs = log_orbitals.shape[-1]

        if not self.laguerre:
            if self.vary_sigma:
                sigma = self.param(
                    "exp_sigma", 
                    normal_initializer(mean=1.0, stddev=1e-3), 
                    () if self.single_sigma else (n_orbs,), 
                    self.param_dtype
                )
                # sigma = sigma*0 + 1.0
                # add bc dimension for sdim
                if self.single_sigma:
                    sigma = sigma[...,None]
            else:
                sigma = jnp.ones((n_orbs,))
            exp_arg = jnp.sum(sigma*(x[...,None])**2, axis=-2) # bc log_orbitals and sum sdim
            log_orbitals += -0.5*exp_arg # logspace
            # sigmas = tuple(np.linspace(0.5, 2, 5))
            # log_orbitals += GaussianEnvelopes(sigmas, param_dtype=self.param_dtype)(x)[..., None]

        if self.hartree_fock:
            mo_coeff_init = jnp.eye(n_orbs, n_particles_max)
            assert mo_coeff_init.shape == (n_orbs, n_particles_max)
            mo_coeff = make_complex_params(
                self, 
                "mo_coeff", 
                array_init(mo_coeff_init, noise=1e-3), 
                (n_orbs, n_particles_max), 
                dtype=self.param_dtype, 
                complex_params=self.complex_coeff
            )
            # make it hartree fock, but do it in LOGSPACE (!!!)
            # vmap over the orbitals to create
            def _make_log_mo_orbs(cs):
                return logsumexp_cplx(log_orbitals, b=cs, axis=-1)
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
    backflow: Callable = None
    orbital_backflow: Callable = None
    ee_cusp: bool = False
    ee_cusp_alpha: float = 10.
    trainable_cusp: bool = True
    scale: float = None
    rc_feat: float = 5.0
    full_det: bool = False
    additive_bf: bool = True
    jastrow_basis: bool = False
    vary_scale: bool = True

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-1]
        N = sum(self.n_per_spin)
        x = x.reshape(-1, N, self.sdim)
        si = np.array([+1]*self.n_per_spin[0] + [-1]*self.n_per_spin[1])

        # if self.scale is not None:
        #     x *= self.scale

        # _, rij_dist_raw = get_distance_matrix(x)
        # rij_dist_raw = rij_dist_raw[...,None] # consistency

        # scales = self.param("scale", normal_initializer(mean=1.0 if self.scale is None else self.scale, stddev=1e-3), (self.sdim,), self.param_dtype)


        # ON THE UNSCALED SYSTEM IF IT NEEDS TO BE HOLOMORPHIC!!!
        rij_diff, rij_dist = get_distance_matrix(x)
        # SHOULD I SQUARE THIS???
        rij_dist = rij_dist[...,None]
        # additional features: distance from the center (cannot contain the scale, since it's no longer holo otherwise)
        ri = x
        ri_norm = jnp.linalg.norm(ri, axis=-1, keepdims=True)**2 # only the square is differentiable everywhere

        #########
        x_scaled = x.copy()
        if self.vary_scale:
            scales = self.param("scale", normal_initializer(mean=1.0 if self.scale is None else self.scale, stddev=1e-3), (1,), self.param_dtype)
        else:
            scales = 1.0 if self.scale is None else self.scale
        x_scaled *= scales      

        if self.backflow is not None:
            x_bf = self.backflow(ri, rij_dist, rij_diff=rij_diff, si=si, ri_norm=ri_norm)
            x_slater = x_scaled+x_bf
        else:
            x_slater = x_scaled #jnp.zeros_like(x)

        if self.orbital_backflow is None:
            orb_bf = None
        else:
            orb_bf = self.orbital_backflow(ri, rij_dist, rij_diff=rij_diff, si=si, ri_norm=ri_norm)

        log_psi = LogSlaterDet(n_per_spin=self.n_per_spin, orbitals=self.orbitals, take_exp=True, full_det=self.full_det)(x_slater, ri=x, orbital_backflow=orb_bf)

        if self.ee_cusp: # we will use the original coordinates here (!)
            # default_cusp_scale_sqrt = jnp.sqrt(self.ee_cusp_alpha)
            default_cusp_scale = self.ee_cusp_alpha
            # separate for each spin
            # if self.trainable_cusp:
            #     cusp_scale_sqrt = self.param("cusp_scale_ee_sqrt", jax.nn.initializers.constant(default_cusp_scale_sqrt), (2,), self.param_dtype)
            # else:
            #     cusp_scale_sqrt = default_cusp_scale_sqrt
            cusp_scale = self.param("cusp_scale_ee", jax.nn.initializers.constant(default_cusp_scale), (), self.param_dtype) # holomorphic
            cusp_scale = jnp.array([cusp_scale, cusp_scale])
            log_psi += el_el_cusp(rij_dist.squeeze(-1), self.n_per_spin, alpha=cusp_scale)  # keep positive
            # log_psi += el_el_cusp(rij_dist_raw.squeeze(-1), self.n_per_spin, alpha=cusp_scale_sqrt**2)  # keep positive

        if self.jastrow is not None:
            # WATCH OUT, WE WILL OVERWRITE THINGS

            if self.jastrow_basis: # meaning we need to define a localized basis
                assert False, "shouldn't reach this point (!)"
                rc_feat = self.rc_feat # is scaled (!)
                square = self.ee_cusp
                rij_diff = get_tanh_feature(rij_diff, rc=rc_feat, square=square, flatten=True, keep_sign=True)
                rij_dist = get_tanh_feature(rij_dist, rc=rc_feat, square=square, flatten=True)
                ri = get_tanh_feature(ri, rc=rc_feat, square=False, flatten=True)
                ri_norm = get_tanh_feature(ri_norm, rc=rc_feat, square=True, flatten=True) # if not squared: not differentiable

            jastrow = make_list(self.jastrow)
            assert len(jastrow) in (1, 2)  # complex
            J = 0.            
            for jastrow_fn, c in zip(jastrow, [1, 1j*2*np.pi]):
                J = J + c*jastrow_fn(
                    rij_dist, si=si, rij_diff=rij_diff, ri=ri, ri_norm=ri_norm, x_scaled=x_scaled
                )
            log_psi = log_psi + J

        return log_psi.reshape(*batch_shape)


def make_list(x):
    if hasattr(x, "__len__"):
        return list(x)
    else:
        return [x]

class SimplestJastrow(nn.Module):
    param_dtype: DType = float
    embedding_dim: int = 4
    hidden_dim: int = None
    pooling: Callable = jnp.mean

    @nn.compact
    def __call__(self, rij_dist, rij_diff=None, ri=None, ri_norm=None, si=None, x_scaled=None):

        hidden_dims = (self.hidden_dim,) if self.hidden_dim is not None else ()

        # print("rij, Rij = ", rij_dist.shape, Rij_dist.shape)
        n_particles = rij_dist.shape[-2]
        rij_dist = pick_triu(rij_dist, k=1, has_aux=True)        
        if rij_diff is not None:
            rij_diff = pick_triu(rij_diff, k=1, has_aux=True)       
        if ri_norm is not None and ri is not None:
            assert ri_norm.shape[:-1] == ri.shape[:-1]
        if si is not None:
            assert si.size == n_particles
            sij = get_spin_spin_matrix(si)
            sij = pick_triu(sij, k=1, has_aux=False)

        contributions = []

        xi_feat = []
        if ri is not None:
            xi_feat.append(ri)
        if ri_norm is not None:
            xi_feat.append(ri_norm)
        if len(xi_feat) > 0:
            xi_feat = jnp.concatenate(xi_feat, axis=-1)
            phi_ri = nk.nn.blocks.MLP(
                output_dim=self.embedding_dim,
                hidden_dims=hidden_dims,
                hidden_activations=DEFAULT_HIDDEN_ACTIVATION,
                output_activation=DEFAULT_OUTPUT_ACTIVATION,
                kernel_init=DEFAULT_KERNEL_INIT,
                param_dtype=self.param_dtype,
                use_output_bias=True,
                name="Jast_ri",
            )(xi_feat)
            contributions.append(phi_ri)

        # build a DeepSet by hand

        xij_feat = [rij_dist]
        if rij_diff is not None:
            assert rij_diff.shape[:-1] == rij_dist.shape[:-1]
            xij_feat.append(rij_diff)
        if si is not None:
            sij = jnp.broadcast_to(sij[..., None], rij_dist.shape)
            xij_feat.append(sij)
        xij_feat = jnp.concatenate(xij_feat, axis=-1)

        phi_rij = nk.nn.blocks.MLP(
            output_dim=self.embedding_dim,
            hidden_dims=hidden_dims,
            hidden_activations=DEFAULT_HIDDEN_ACTIVATION,
            output_activation=DEFAULT_OUTPUT_ACTIVATION,
            kernel_init=DEFAULT_KERNEL_INIT,
            param_dtype=self.param_dtype,
            use_output_bias=True,
            name="Jast_rij",
        )(xij_feat)
        contributions.append(phi_rij)
        
        # pool
        phis = jnp.concatenate(contributions, axis=-2)
        phi = self.pooling(phis, axis=-2)

        J = nk.models.MLP(
            hidden_dims=hidden_dims,
            hidden_activations=DEFAULT_HIDDEN_ACTIVATION,
            output_activation=DEFAULT_OUTPUT_ACTIVATION,
            kernel_init=DEFAULT_KERNEL_INIT,
            param_dtype=self.param_dtype,
            use_output_bias=True,
            name="Jast_Rho"
        )(phi)

        return J
    

class SimpleBackFlow(nn.Module):
    sdim: int
    hidden_dim: int = 4
    param_dtype: Any = float
    per_dim: bool = False
    add_global_info: bool = False
    embedding_dim: int = 4
    complex_proj: bool = False

    @nn.compact
    def __call__(self, ri, rij_dist, rij_diff=None, si=None, ri_norm=None):
        """
        Args:
            ri: [..., N, D]
            rij_diff: [..., N, N, D]
            rij_dist: [..., N, N, D']
        
        Returns:
            xb: [..., N, sdim]
        """

        sdim = self.sdim
        n_particles = ri.shape[-2]

        assert rij_diff.shape[-3] == rij_diff.shape[-2]
        assert rij_diff.shape[-2] == n_particles
        assert rij_diff.shape[-1] == ri.shape[-1]
        assert rij_dist.shape[-3] == rij_dist.shape[-2]
        assert rij_dist.shape[-2] == n_particles
        if ri_norm is not None:
            assert ri_norm.shape[:-1] == ri.shape[:-1]
        
        hidden_dims = (self.hidden_dim,) if self.hidden_dim is not None else None

        #############################
        # pair contributions        #
        #############################
        pair_contr = 0.

        # ELECTRON-ELECTRON interaction
        # avoid taking nasty norms
        rij_dist = remove_diag(rij_dist, has_aux_axis=True)
        if rij_diff is not None:
            rij_diff = remove_diag(rij_diff, has_aux_axis=True)

        ###

        xi_feat = [ri]
        if ri_norm is not None:
            xi_feat.append(ri_norm)
        # if np.sum(si) != 0:
        #     xi_feat.append(jnp.broadcast_to(si[...,None], ri_norm.shape))
        xi_feat = jnp.concatenate(xi_feat, axis=-1)

        xij_feat = [rij_dist]
        if si is not None:
            sij = get_spin_spin_matrix(si)
            sij = remove_diag(sij, has_aux_axis=False)
            xij_feat.append(jnp.broadcast_to(sij[...,None], rij_dist.shape))
        if rij_diff is not None:
            xij_feat.append(rij_diff)
        xij_feat = jnp.concatenate(xij_feat, axis=-1)
        
        if self.add_global_info:
            d = 4 if self.hidden_dim is None else self.hidden_dim
            xi_feat_g = nk.nn.blocks.MLP(
                output_dim=d,
                hidden_dims=hidden_dims,
                param_dtype=self.param_dtype,
                hidden_activations=DEFAULT_HIDDEN_ACTIVATION,
                output_activation=DEFAULT_HIDDEN_ACTIVATION,
                kernel_init=DEFAULT_KERNEL_INIT,
                use_output_bias=True,
                name="simp_bf_enc"
            )(xi_feat)
            xij_feat_g = jnp.broadcast_arrays(xi_feat_g[...,:,None,:], xi_feat_g[...,None,:,:])
            xij_feat_g = jnp.concatenate(xij_feat_g, axis=-1)
            xij_feat_g = remove_diag(xij_feat_g, has_aux_axis=True)
            
            xij_feat = jnp.concatenate([xij_feat, xij_feat_g], axis=-1) # add the global info

        net_ee = nk.nn.blocks.MLP(
            output_dim=self.embedding_dim,
            hidden_dims=hidden_dims,
            param_dtype=self.param_dtype,
            hidden_activations=DEFAULT_HIDDEN_ACTIVATION,
            output_activation=DEFAULT_HIDDEN_ACTIVATION,
            kernel_init=DEFAULT_KERNEL_INIT,
            use_output_bias=True,
            name="simp_bf_ee"
        )(xij_feat)
        # direction from diff, sum over other particles
        
        # project back
        W = make_complex_params(self, "Wproj", DEFAULT_KERNEL_INIT, (net_ee.shape[-1], sdim), dtype=self.param_dtype, complex_params=self.complex_proj)
        contr = net_ee @ W

        # pool over other particles
        pair_contr = pair_contr + jnp.sum(contr, axis=-2)

        #############################
        # single contributions      #
        #############################
        # this does not decay exponentially now (but the orbitals do!)
        # to handle the electric external field
        shift = self.param(
            "simp_bf_shift",
            jax.nn.initializers.zeros,
            (sdim,),
            self.param_dtype,
        )
        single_contr = shift

        x_bf = pair_contr + single_contr

        return x_bf

class AttentionBackflow(nn.Module):
    hidden_dim: int = 4
    param_dtype: Any = float
    n_layers: int = 1
    project_back: bool = True
    project_first: bool = True
    square: bool = False
    complex_proj: bool = False
    concat_xi: bool = False
    add_shift: bool = True
    kernel_init: Callable = DEFAULT_KERNEL_INIT

    @nn.compact
    def __call__(self, ri, rij_dist, ri_norm=None, rij_diff=None, si=None):
        """ Does not depend on the ee interaction
        Args:
            ri: [..., N, D]
            rij_diff: [..., N, N, D]
            rij_dist: [..., N, N, D']
        
        Returns:
            xb: [..., N, sdim]
        """
        # TODO: remove rii components (!)

        sdim = ri.shape[-1]
        n_particles = ri.shape[-2]

        assert rij_diff.shape[-3] == rij_diff.shape[-2]
        assert rij_diff.shape[-2] == n_particles
        assert rij_diff.shape[-1] == ri.shape[-1]
        assert rij_dist.shape[-3] == rij_dist.shape[-2]
        assert rij_dist.shape[-2] == n_particles
        assert ri_norm.shape[:-1] == ri.shape[:-1]
        
        assert rij_dist.shape[-1] == 1, f"don't use bases {rij_dist.shape}"
        assert ri_norm.shape[-1] == 1, f"don't use bases {ri_norm.shape}"

        rij_dist = rij_dist.squeeze(-1)
        ri_norm = ri_norm.squeeze(-1)
        if self.square:
            scale = rij_dist
            rij_dist *= scale
            rij_diff *= scale[...,None]

        ###############################
        # NODE CONFIGURATIONS
        xi = [ 
            ri_norm[...,None],
            ri
        ]
        xi = jnp.concatenate(xi, axis=-1)

        ###############################
        # EDGE CONFIGURATIONS
        # rij_dist = remove_diag(rij_dist, has_aux_axis=False)
        # rij_diff = remove_diag(rij_diff, has_aux_axis=True)

        xij = [
            rij_dist[...,None],
            rij_diff,
        ]
        if si is not None:
            assert si.size == n_particles
            sij = get_spin_spin_matrix(si)
            sij = jnp.broadcast_to(sij, rij_dist.shape)
            xij.append(sij[...,None])
        xij = jnp.concatenate(xij, axis=-1)

        ###############################

        if self.project_first:
            Wpi = make_complex_params(self, "proj_Wi", self.kernel_init, (xi.shape[-1], self.hidden_dim), self.param_dtype, complex_params=False)
            xi = xi @ Wpi
            Wpij = make_complex_params(self, "proj_Wij", self.kernel_init, (xij.shape[-1], self.hidden_dim), self.param_dtype, complex_params=False)
            xij = xij @ Wpij
        
        xi0 = xi
        xij0 = xij
        for i_att in range(self.n_layers):
            xi, xij, mij = Attention(hidden_dim=self.hidden_dim, particle_attention=True, kernel_init=self.kernel_init, param_dtype=self.param_dtype, complex_params=False, concat_xi=self.concat_xi, name=f"att{i_att}")(xi, xij)
            # xij are not yet adapted
            if self.n_layers > 1:
                # need an extra processing the xij if we need them in a next layer
                nn_input = jnp.concatenate([xij, mij], axis=-1) # unstable?
                xij = nk.nn.blocks.MLP(output_dim=xij.shape[-1], param_dtype=self.param_dtype, hidden_dims=(self.hidden_dim,), hidden_activations=DEFAULT_HIDDEN_ACTIVATION, kernel_init=self.kernel_init)(nn_input)
                xi = jnp.concatenate([xi, xi0], axis=-1)
                xij = jnp.concatenate([xij, xij0], axis=-1)

        if self.project_back:
            Wp = make_complex_params(self, "proj_Wp", self.kernel_init, (xi.shape[-1], sdim), self.param_dtype, complex_params=self.complex_proj)
            xi = xi @ Wp

        if self.add_shift:
            shift = make_complex_params(
                self,
                "bf_shift",
                jax.nn.initializers.zeros,
                (xi.shape[-1],),
                self.param_dtype,
                complex_params=self.complex_proj
            )
            xi += shift

        return xi




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

    # rij_dist, si=si, rij_diff=rij_diff, ri=ri, ri_norm=ri_norm, x_scaled=x_scaled
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

        if self.laguerre:
            mus = get_mus(self.n_max, Ecutoff=self.Ecutoff, sdim=sdim, m_constraint=False)
            assert mus.shape[0] >= n_particles_max, \
                f"need higher n_max: {len(mus)}"
            assert mus.shape[-1] == 3
            basis_single_body = compute_laguerre_orbitals(x_scaled, mus)
        else:
            ns = np.array(get_ns_list(sdim, self.n_max))
            if self.Ecutoff is not None:
                # print("Cutting off from", ns.shape)
                Es = ns.sum(axis=-1) + sdim/2
                idx_cutoff = Es <= self.Ecutoff + 1e-9 # rounding hack
                ns = ns[idx_cutoff,:]
                # print("To", ns.shape, "with max n = ", np.max(ns))
            max_n_in_basis = np.max(ns)
            hermites = hermite_till_nmax(x_scaled, n_max=max_n_in_basis)
            basis_single_body = jax.vmap(hermite_comb_nd, in_axes=(None, 0), out_axes=-1)(hermites, ns)
        n_basis_single = basis_single_body.shape[-1]

        if not self.laguerre:
            if self.vary_sigma:
                sigma = self.param(
                    "exp_sigma_jastrow", 
                    normal_initializer(mean=1.0, stddev=1e-3), 
                    () if self.single_sigma else (n_basis_single,), 
                    self.param_dtype
                )
                # sigma = sigma*0 + 1.0
                # add bc dimension for sdim
                if self.single_sigma:
                    sigma = sigma[...,None]
            else:
                sigma = jnp.ones((n_basis_single,))            
            exp_arg = jnp.sum(sigma*(x[...,None])**2, axis=-2) # bc basis and sum sdim
            basis_single_body *= jnp.exp(-0.5*exp_arg) # real space
            # sigmas = tuple(np.linspace(0.5, 2, 5))
            # basis_single_body += GaussianEnvelopes(sigmas, param_dtype=self.param_dtype)(x)[...,None]

        # now combine all these bases
        basis = jax.vmap(basis_combine_jastrow)(basis_single_body)

        n_basis = basis.shape[-1]
        # sum over particle pairs BEFORE HF
        basis = basis.mean(axis=-2)

        jastrow_coeff = make_complex_params(
            self,
            "jastrow_coeff",
            jax.nn.initializers.normal(stddev=1e-5),
            (n_basis,), # just project to a scalar (!)
            dtype=self.param_dtype,
            complex_params=self.complex_coeff
        )

        J = basis.dot(jastrow_coeff)
        # # sum over particle pairs
        # J = jnp.mean(J, axis=-1)
        return J # logspace  
    
# try later
class GaussianEnvelopes(nn.Module):
    # fixed scale, but moving linear coeficients
    # done in logspace
    sigmas: Tuple[float,...]
    param_dtype: DType = jnp.float64
    @nn.compact
    def __call__(self, x):
        sigmas = jnp.array(self.sigmas)
        exp_args = jnp.sum(-0.5*(x[...,None]/sigmas)**2, axis=-2) # sum over sdim, last dimension is sigmas axis
        coeffs = self.param("gauss_env_coeffs", normal_initializer(mean=1), (sigmas.size,), self.param_dtype)
        assert exp_args.shape[-1] == coeffs.size
        return logsumexp_cplx(exp_args, b=coeffs, axis=-1)

@jax.jit
def basis_combine_jastrow(single_body_basis): # in LOGSPACE(!)
    assert single_body_basis.ndim == 2, f"should be single sample, got {single_body_basis.shape}, maybe forgot to vmap over the batch size?"
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
    chi_munu_ij = jax.vmap(lambda a, b: jnp.ravel(a*b))(*chi_munu_ij) # product over mu and nu, in real space
    # output is now of dimension [i<j, mu*nu] # last axis must later be summed over
    assert chi_munu_ij.shape[-1] == Nb*Nb, f"got {chi_munu_ij.shape} vs Nb={Nb}"
    assert chi_munu_ij.shape[0] == int(N*(N-1)/2), f"got {chi_munu_ij.shape} vs N={N}"

    return chi_munu_ij



class BasisOrbitalBackflow(nn.Module):
    # log orbitals
    n_orbitals: int
    n_max: int
    n_per_spin: Tuple[int,int]
    single_sigma: bool = False
    param_dtype: DType = jnp.float64
    complex_coeff: bool = True
    Ecutoff: int = None
    vary_sigma: bool = True
    laguerre: bool = True


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

        if self.laguerre:
            mus = get_mus(self.n_max, Ecutoff=self.Ecutoff, sdim=sdim, m_constraint=False)
            # assert mus.shape[0] >= n_particles_max, \
            #     f"need higher n_max: {len(mus)}"
            assert mus.shape[-1] == 3
            basis_single_body = compute_laguerre_orbitals(x, mus)
        else:
            ns = np.array(get_ns_list(sdim, self.n_max))
            if self.Ecutoff is not None:
                # print("Cutting off from", ns.shape)
                Es = ns.sum(axis=-1) + sdim/2
                idx_cutoff = Es <= self.Ecutoff + 1e-9 # rounding hack
                ns = ns[idx_cutoff,:]
                # print("To", ns.shape, "with max n = ", np.max(ns))
            max_n_in_basis = np.max(ns)

            hermites = hermite_till_nmax(x, n_max=max_n_in_basis)
            basis_single_body = jax.vmap(hermite_comb_nd, in_axes=(None, 0), out_axes=-1)(hermites, ns)
        n_basis_single = basis_single_body.shape[-1]

        if not self.laguerre:
            if self.vary_sigma:
                sigma = self.param(
                    "exp_sigma_backflow", 
                    normal_initializer(mean=1.0, stddev=1e-3), 
                    () if self.single_sigma else (n_basis_single,), 
                    self.param_dtype
                )
                # sigma = sigma*0 + 1.0
                # add bc dimension for sdim
                if self.single_sigma:
                    sigma = sigma[...,None]
            else:
                sigma = jnp.ones((n_basis_single,))   
            exp_arg = jnp.sum((sigma*x[...,None])**2, axis=-2) # bc basis and sum sdim
            basis_single_body *= jnp.exp(-0.5*exp_arg) # real space

        # now combine all these bases
        basis = jax.vmap(basis_combine_backflow)(basis_single_body)

        n_basis = basis.shape[-1]
        # sum over OTHER particles before
        basis = remove_diag(basis, has_aux_axis=True) # remove diag from axis 1 and 2 (has mu*nu axis)
        basis = basis.mean(axis=-2) # sum over the other particles

        backflow_coeff = make_complex_params(
            self,
            "backflow_coeff",
            jax.nn.initializers.normal(stddev=1e-2),
            (n_basis, self.n_orbitals), # just project to a scalar (!)
            dtype=self.param_dtype,
            complex_params=self.complex_coeff
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


class HoloOrbitalBackflow(nn.Module):
    n_orbitals: int
    n_per_spin: Tuple[int,int]
    param_dtype: DType = jnp.float64
    hidden_dims: int = 4

    @nn.compact
    def __call__(self, ri, rij_dist, ri_norm=None, rij_diff=None, si=None):
        n_particles = sum(self.n_per_spin)

        # we only work with ri
        x = ri
        assert x.shape[-2] == n_particles
        sdim = x.shape[-1]
        x = x.reshape(-1, n_particles, sdim)

        xij = jnp.concatenate(jnp.broadcast_arrays(x[...,:,None,:], x[...,None,:,:]), axis=-1)
        xij = remove_diag(xij, has_aux_axis=True)

        y = nknn.blocks.MLP(
            output_dim=self.n_orbitals, 
            hidden_dims=(self.hidden_dims,), 
            param_dtype=self.param_dtype, 
            hidden_activations=poly_logcosh, 
            kernel_init=lecun_normal_scale(std=1e-1),
            use_hidden_bias=False, # these only give us noise
            use_output_bias=False,
        )(xij)
        y = y.mean(axis=-2) # sum over the other particles
        return y
    
# class ClassicalBackflow(nn.Module):
#     param_dtype: DType = jnp.float64
#     n_bases: int = 4