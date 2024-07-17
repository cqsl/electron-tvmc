# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import netket as nk

nk.config.netket_experimental_fft_autocorrelation = True

# %load_ext autoreload
# %autoreload 2

import numpy as np
import netket as nk
import netket.experimental as nkx
import jax
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
from functools import partial
import json
import optax
import pickle
import matplotlib.pyplot as plt
from src.utils import tree_size_cumsum, is_complex_dtype, real_dtype, mpi_shuffle_along_axis
import warnings
from src.quantumdot.potential import quantum_dot_V0, quantum_dot_Vint


import os, sys

from netket.utils import mpi
import netket
from netket.utils.mpi import n_nodes, mpi_sum_jax
from netket.utils.mpi import mpi_sum as mpi_sum_np


# +
# this is an example script of the quantum dot model
# this file is a great simplification of the original scripts and 
# intended for (1) illustrative purposes, and (2) providing reproducible subroutines
# so if anything is missing or deviating from the manuscript, please contact us 
# -

DTYPE = jnp.complex128

# +
n_per_spin = (6, 0)
sdim = 2
N = sum(n_per_spin)

kappa_0 = 1.0
omega = 1.0

rs = 1.0
effective_scale = np.sqrt(N)
hi = nk.hilbert.Particle(n_per_spin, D=sdim, pbc=False)

# +
from src.quantumdot.model import (
    HOOrbs, HOModel, 
    BasisJastrow, BasisOrbitalBackflow,
)

# take low numbers to run locally
n_max = 4
Ecutoff = 4

orbitals = HOOrbs(
    n_max, 
    n_per_spin=n_per_spin, 
    param_dtype=DTYPE, 
    Ecutoff=Ecutoff, 
)

n_max_jastrow = 2
Ecutoff_jastrow = 4
ma_jastrow = BasisJastrow(
    n_max=n_max_jastrow,
    n_per_spin=n_per_spin,
    param_dtype=DTYPE,
    Ecutoff=Ecutoff_jastrow,
)

n_max_backflow = 2
Ecutoff_backflow = 4
ma_backflow = BasisOrbitalBackflow(
    n_orbitals=max(n_per_spin),
    n_max=n_max_backflow,
    n_per_spin=n_per_spin,
    param_dtype=DTYPE,
    Ecutoff=Ecutoff_backflow,
)

ma = HOModel(
    orbitals,
    n_per_spin,
    sdim,
    param_dtype=DTYPE,
    jastrow=ma_jastrow,
    orbital_backflow=ma_backflow,
)
print("MODEL:")
print(ma)

# +
from src.sampler import MetropolisGaussAdaptive

sigma = 0.16
sa_fn = partial(MetropolisGaussAdaptive, target_acceptance=0.6, initial_sigma=sigma)
sa = sa_fn(hi, n_chains=64, n_sweeps=hi.size*5, dtype=real_dtype(DTYPE))

print("SAMPLER = ", sa)
# -

n_samples = 1024 # higher in practice
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=4, chunk_size=16)


print("n_parameters:", vs.n_parameters)

if mpi.node_number == 0:
    print(jax.tree_util.tree_map(lambda x: f"{x.shape} = {x.size} : {x.dtype}", vs.parameters))


# +
from src.operator import KineticEnergy

rs = 1.0
omega = 1.0
pot_fn_0 = partial(quantum_dot_V0, sdim=sdim, omega2=omega**2, m=rs**2)
epot_0 = nk.operator.PotentialEnergy(hi, pot_fn_0)
pot_fn_int = partial(quantum_dot_Vint, sdim=sdim, kappa=(kappa_0/rs))
epot_int = nk.operator.PotentialEnergy(hi, pot_fn_int)
ekin = KineticEnergy(hi, mass=rs**2)

ham = ekin + epot_0 + epot_int

# +
total_steps = 10 # higher in practice
lr_schedule = optax.piecewise_constant_schedule(
    init_value=1e-1, 
    boundaries_and_scales={int(total_steps*0.3):0.2, int(total_steps*0.6):0.2, int(total_steps*0.95):0.2}
)
print("lr_schedule:", lr_schedule)

opt = nk.optimizer.Sgd(lr_schedule)
qgt = nk.optimizer.qgt.QGTJacobianPyTree(diag_shift=1e-5, holomorphic=True)
sr = nk.optimizer.SR(qgt)

gs = nk.driver.VMC(ham, opt, variational_state=vs, preconditioner=sr)
gs


# +
from src.quantumdot.model import get_ns_list

Efree = 0.
for ns in n_per_spin:
    ns = np.array(get_ns_list(sdim, n_max))[:ns,:]
    Efree += ns.sum() + sdim/2
print("Free QHO energy = ", Efree, Efree*omega)

# -

print("n_parameters:", vs.n_parameters)

gs.run(total_steps, out='vmc')

vs.n_samples = 1024 #512*1024
print("n_samples for tevo:", vs.n_samples)

# run hot
for i_hot in range(10):
    print("i_hot = ", i_hot)
    vs.reset()
    vs.samples
    print("ACCEPTANCE:", vs.sampler_state.acceptance, flush=True)
print("Final sampler_scale:", vs.sampler_state.rule_state, flush=True)


E0expect = vs.expect(ham)


dt = 1e-2
# +
kappa_t = 2.0

kappa_t, rs_t, omega_t, effective_scale_t = interpret_kappa(kappa_t)

Tmax = 1.0

ekin_t = KineticEnergy(hi, mass=rs**2)
epot_fn_0_t = partial(quantum_dot_V0, sdim=sdim, omega2=omega_t**2, m=rs**2)
epot_0_t = nk.operator.PotentialEnergy(hi, epot_fn_0_t) 
pot_fn_int_t = partial(quantum_dot_Vint, sdim=sdim, kappa=(kappa_t/rs))
epot_int_t = nk.operator.PotentialEnergy(hi, pot_fn_int_t) 
ham_obj_t = ekin_t + epot_0_t + epot_int_t

def ham_t(t):
    return ham_obj_t

# +
# from netket.experimental.driver import TDVP
from src.solver import smooth_svd
from functools import partial
from netket.experimental.dynamics import RK45, Heun, RK23

# Create integrator for time propagation

integrator = Heun(
    dt=float(dt),
)

print("Integrator = ", integrator)

qgt = nk.optimizer.qgt.QGTJacobianPyTree(holomorphic=True, diag_shift=0, diag_scale=0)
solver = partial(smooth_svd, acond=1e-8, rcond=1e-14)
print("QGT = ", qgt)


# Quenched hamiltonian: this has a different transverse field than `ha`
te = nkx.driver.TDVP(
    ham_t,
    variational_state=vs,
    integrator=integrator,
    t0=0,
    qgt=qgt,
    propagation_type='real',
    error_norm="qgt",
    linear_solver=solver
)

# +
print("Gradient:", vs.expect_and_grad(ham_t(0)))


# vs.chunk_size = 1
te.run(
    T=Tmax,
    out='tdvp',
    show_progress=True,
)
# -




