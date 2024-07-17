# +

# example script for the  Harmonic interaction (non-holomorphic, real parameter case)

import netket as nk
nk.config.netket_experimental_fft_autocorrelation = True
import jax
import jax.numpy as jnp
import netket as nk
import flax
import flax.linen as nn
from typing import Any
import os, sys
import copy
import netket.nn as nknn

import matplotlib.pyplot as plt
import numpy as np
import json
# -

# back
N = 30
EXACT = True # which model to take

Gint = 1.0
hi = nk.hilbert.Particle(N=N, D=1, pbc=False)

from src.sampler import MetropolisGaussAdaptive
sampler = MetropolisGaussAdaptive(
    hi,
    n_chains=72 * 8,
    sweep_size=10 * hi.size,
    target_acceptance=0.6,
    initial_sigma=0.01,
    dtype=jnp.float64,
)

# +
def vpot(x, omega=1.0, g=1.0):
    assert x.ndim == 1
    x = x.reshape(N)
    dis = -x[None, :] + x[:, None]
    idx = jnp.triu_indices(N, k=1)
    dis = dis[idx[0], idx[1]]
    interact = (
        0.5 * g * (dis**2).sum(axis=-1)
    )  # sum over dimensions and particle pairs
    harm = (
        0.5 * omega**2 * (x**2).sum(axis=-1)
    )  # jnp.sum(jnp.linalg.norm(x, axis=-1)**2)
    return harm + interact

from src.operator import KineticEnergy
ekin = KineticEnergy(hi, mass=1.0)
pot = nk.operator.PotentialEnergy(hi, lambda x: vpot(x, omega=1.0, g=Gint))
ham_0 = ekin + pot
# -

# omega in the paper
# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.61.033613
effective_alpha = jnp.sqrt(1 + N * Gint)

# +

@jax.jit
def _compute_log_gs(x, alpha, beta, phase=0):
    # use 1d (forget dimension)
    x = x.reshape(-1, N)
    idx = jnp.triu_indices(N, k=1)
    dis = -x[:, None, :] + x[:, :, None]
    xij = dis[:, idx[0], idx[1]]

    log_xij = jnp.log(xij + 0j)
    log_det = jnp.sum(log_xij, axis=-1)  # product over pairs

    r2 = x**2
    J = alpha * r2
    J = J.sum(axis=-1) / 2  # sum particles (product)

    R2 = x.sum(axis=-1) ** 2
    K = beta * R2 / 2

    log_psi = log_det - J - K - 1j * phase

    return log_psi


class HarmIntModelExact(nn.Module):
    hilbert: Any
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        scale = self.param("L", jax.nn.initializers.ones, (), self.param_dtype)
        scale = jnp.abs(scale)
        
        alpha_r = self.param(
            "alpha_re",
            jax.nn.initializers.constant(effective_alpha),
            (),
            self.param_dtype,
        )
        alpha_i = self.param(
            "alpha_im", jax.nn.initializers.zeros, (), self.param_dtype
        )
        beta_r = self.param(
            "beta_re",
            jax.nn.initializers.constant((1 - effective_alpha) / N),
            (),
            self.param_dtype,
        )
        beta_i = self.param("beta_im", jax.nn.initializers.zeros, (), self.param_dtype)
        phase = self.param("phase", jax.nn.initializers.zeros, (), self.param_dtype)

        alpha = alpha_r + 1j * alpha_i
        beta = beta_r + 1j * beta_i
        log_psi = _compute_log_gs(x * scale, alpha, beta, phase=phase)

        return log_psi


class HarmIntModel(nn.Module):
    hilbert: Any
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        scale = self.param("L", jax.nn.initializers.ones, (), self.param_dtype)
        scale = jnp.abs(scale)

        alpha_r = self.param(
            "alpha_re",
            jax.nn.initializers.constant(effective_alpha),
            (),
            self.param_dtype,
        )
        alpha_i = self.param(
            "alpha_im", jax.nn.initializers.zeros, (), self.param_dtype
        )
        alpha = alpha_r + 1j * alpha_i

        beta_r = 0
        beta_i = 0 
        beta = beta_r + 1j * beta_i

        log_psi = _compute_log_gs(x * scale, alpha, beta)

        x = x.reshape(-1, N, 1)
        log_psi -= nk.models.DeepSetMLP(
            features_phi=4,
            features_rho=4,
            param_dtype=self.param_dtype,
            hidden_activation=nknn.elu,
            pooling=jnp.mean,
        )(x)
        log_psi -= 1j * nk.models.DeepSetMLP(
            features_phi=4,
            features_rho=4,
            param_dtype=self.param_dtype,
            hidden_activation=nknn.elu,
            pooling=jnp.mean,
        )(x)
        return log_psi


# -

if EXACT:
    model = HarmIntModelExact(hi)
else:
    model = HarmIntModel(hi)

n_samples = 72 * 1024
# n_samples = 1024 # to check
vs = nk.vqs.MCState(
    sampler, model, n_samples=n_samples, n_discard_per_chain=4, chunk_size=16
)

print(
    json.dumps(
        jax.tree.map(lambda x: f"{x.shape} [{x.dtype}]", vs.parameters), indent=4
    ),
    flush=True,
)

print("n_parameters = ", vs.n_parameters)

orig_params = copy.deepcopy(vs.parameters)

print("start running hot", flush=True)
for _ in range(10):
    vs.reset()
    vs.samples
print("finished running hot", flush=True)

opt = nk.optimizer.Sgd(4e-3)
qgt = nk.optimizer.qgt.QGTJacobianPyTree
sr = nk.optimizer.SR(qgt, diag_shift=1e-6, diag_scale=1e-6, holomorphic=False)

gs = nk.VMC(ham_0, opt, variational_state=vs, preconditioner=sr)
print("start vmc", flush=True)
n_runs = 1000
gs.run(n_iter=n_runs, out="vmc")
print("finished vmc", flush=True)


with open(f"vmc.log", 'r') as f:
    data = json.load(f)
_energy = data["Energy"]["Mean"]["real"]
plt.plot(np.arange(len(_energy)), _energy)
plt.show()

E0 = vs.expect(ham_0).mean
print("GROUND STATE ENERGY:", E0, flush=True)


from src.solver import smooth_svd # medvidovic & sels method
from functools import partial
from netket.experimental.dynamics import RK23, Heun

# +
omega_f = 2.0

@jax.jit
def Lt_fn(t):
    o0 = 1.0
    of = omega_f
    A = (of**2 - o0**2) / 2 / of**2
    C = (of**2 + o0**2) / 2 / of**2
    return jnp.sqrt(A * jnp.cos(2 * of * t) + C)

@jax.jit
def g_func(t):
    Lt = Lt_fn(t)
    return Gint/Lt**4

@jax.jit
def vt(x, t=0, omega=1):
    return vpot(x, omega=omega, g=g_func(t))


# +
from src.operator import PotentialEnergyTD
from functools import partial

print(vs.samples.shape)
dt = 1e-2
Tmax = 20
ekin = KineticEnergy(hi, mass=1.0)
pot = PotentialEnergyTD(hi, partial(vt, omega=omega_f))
ham_obj_t = pot.set_t(0) + ekin 

def ham_t(t):
    ham_obj_t._operators[0].set_t(t)
    return ham_obj_t

# +
@jax.jit
def update_parameters(parameters, t):
    params = jax.tree_map(jnp.copy, parameters)
    o0 = 1.0
    of = omega_f
    A = (of**2 - o0**2) / 2 / of**2
    C = (of**2 + o0**2) / 2 / of**2
    Lt_fn = lambda _t: jnp.sqrt(A * jnp.cos(2 * of * _t) + C)
    Lt = Lt_fn(t)
    # Ft = A*of*jnp.sin(2*of*t)/Lt**2
    Ft = jax.grad(Lt_fn)(t) / (2*Lt) * (-2) # last is correction factor due to our definition of the log_gs where we absorbe it
    # Ft = (
    #     A
    #     * of**3
    #     * jnp.sin(2 * of * t)
    #     / (o0**2 * jnp.sin(of * t) ** 2 + of**2 * jnp.cos(of * t) ** 2)
    # )
    taut = 1 / o0 * jnp.arctan(o0 / of * jnp.tan(of * t))
    if not isinstance(params, dict):
        params = params.unfreeze()
    params["L"] = 1/Lt
    params["alpha_im"] = Ft / Lt**2
    if "phase" in params:
        params["phase"] = E0.real * taut
    return params


def exact_callback(step_nr, log_data, driver):
    t = driver.t
    driver.state.reset()
    params = update_parameters(orig_params, t)
    driver.state.parameters = params
    vs.parameters = params
    driver.state.reset()
    vs.reset()
    return True

# +
from netket.experimental.driver import TDVP

integrator = nk.experimental.dynamics.RK23(
    dt=dt, adaptive=True, atol=1e-5, rtol=1e-5, dt_limits=(dt / 10, dt)
)

qgt = nk.optimizer.qgt.QGTJacobianPyTree(holomorphic=False)
solver = partial(smooth_svd, acond=1e-8, rcond=1e-8)

te = TDVP(
    ham_t,
    variational_state=vs,
    integrator=integrator,
    t0=0.0,
    qgt=qgt,
    propagation_type="real",
    error_norm="qgt",
    linear_solver=solver,
)
cbs = []
if EXACT:
    cbs.append(exact_callback)
te.run(
    T=Tmax,
    out=f"tdvp",
    callback=cbs,
    show_progress=True,
)
