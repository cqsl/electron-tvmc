import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import struct
import netket as nk


from typing import Optional, Any
from netket.sampler import MetropolisSampler, MetropolisRule
import netket.jax as nkjax

from netket.hilbert import ContinuousHilbert
from netket.stats import mean as mpi_mean
import jax
import jax.numpy as jnp
import numpy as np


class GaussianRule(MetropolisRule):
    r"""
    A transition rule acting on all particle positions at once.

    New proposals of particle positions are generated according to a
    Gaussian distribution of width sigma.
    """
    initial_sigma: float
    
    def __init__(self, initial_sigma: float = 1.0):
        self.initial_sigma = initial_sigma

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params,
        key,
    ) -> Optional[Any]:
        sigma = self.initial_sigma
        return sigma

    def transition(rule, sampler, machine, parameters, state, key, r):
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError(
                "Gaussian Rule does not work with complex " "basis elements."
            )

        n_chains = r.shape[0]
        hilb = sampler.hilbert

        pbc = np.array(hilb.n_particles * hilb.pbc, dtype=r.dtype)
        boundary = np.tile(pbc, (n_chains, 1))

        Ls = np.array(hilb.n_particles * hilb.extent, dtype=r.dtype)
        modulus = np.where(np.equal(pbc, False), jnp.inf, Ls)

        sigma = state.rule_state
        prop = jax.random.normal(
            key, shape=(n_chains, hilb.size), dtype=r.dtype
        ) * jnp.asarray(sigma, dtype=r.dtype)

        rp = jnp.where(np.equal(boundary, False), (r + prop), (r + prop) % modulus)

        return rp, None

    def __repr__(self):
        return f"GaussianRule(floating)"


class MetropolisGaussAdaptive(MetropolisSampler):
    target_acceptance: float = None
    sigma_limits: Any = None

    def __init__(self, hilbert, *args, initial_sigma=1.0, sigma_limits=None, target_acceptance=0.6, **kwargs):
        rule = GaussianRule(
            initial_sigma=initial_sigma,
        )
        if sigma_limits is None:
            sigma_limits = (initial_sigma*1e-2, initial_sigma*1e2)
        sigma_limits = tuple(sigma_limits)
        if not isinstance(hilbert, ContinuousHilbert):
            raise ValueError(
                f"This sampler only works for ContinuousHilbert Hilbert spaces, got {type(hilbert)}.")
        super().__init__(hilbert, rule, *args, **kwargs)
        self.target_acceptance = target_acceptance
        self.sigma_limits = sigma_limits

    def _sample_next(self, machine, parameters, state):
        new_state, new_σ = super()._sample_next(machine, parameters, state)

        if self.target_acceptance is not None:
            acceptance = new_state.n_accepted / new_state.n_steps
            sigma = new_state.rule_state
            new_sigma = sigma / (self.target_acceptance / jnp.max(jnp.stack([acceptance, jnp.array(0.05)])))
            new_sigma = jnp.max(jnp.array([new_sigma, self.sigma_limits[0]]))
            new_sigma = jnp.min(jnp.array([new_sigma, self.sigma_limits[1]]))
            new_rule_state = new_sigma
            new_state = new_state.replace(
                rule_state=new_rule_state
            )

        return new_state, new_σ
