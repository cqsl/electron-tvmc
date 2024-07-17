import jax.numpy as jnp
import jax.numpy as jnp
from src.utils import get_full_distance_matrix

REG_EPS = 1e-12


def coulomb_potential(dist_mat, charges=None, smooth=None):
    if smooth is None:
        denom_mat = dist_mat + REG_EPS
    else:
        denom_mat = jnp.sqrt(dist_mat**2 + smooth**2)
    if charges is None:
        numer = 1.0
    else:
        numer = charges
    return numer / denom_mat


def get_el_el_potential_energy(r_el, smooth=None):
    assert r_el.ndim == 2
    n_el = r_el.shape[-2]
    eye = jnp.eye(n_el)
    dist_matrix = get_full_distance_matrix(r_el)
    # add eye to diagonal to prevent div/0
    # E_pot = jnp.triu(1.0 / (dist_matrix + eye + reg_eps), k=1)
    E_pot = jnp.triu(coulomb_potential(dist_matrix + eye, smooth=smooth), k=1)
    return jnp.sum(E_pot, axis=[-2, -1])


def quantum_dot_potential_energy(x, sdim=2, omega2=1., m=1., kappa=1.0, with_coulomb=True):
    assert x.ndim == 1
    x = x.reshape(-1, sdim)
    Eharm = 0.5*m*jnp.sum(omega2*x**2)
    E = Eharm
    if with_coulomb:
        # rij = get_distance_matrix(x)[-1] # that's already done inside the following function (!!!!!!!)
        Ecoul = get_el_el_potential_energy(x)
        E += Ecoul*kappa
    return E

def quantum_dot_V0(x, sdim=2, omega2=1., m=1.):
    assert x.ndim == 1
    x = x.reshape(-1, sdim)
    Eharm = 0.5*m*jnp.sum(omega2*x**2)
    return Eharm

def quantum_dot_Vint(x, sdim=2, kappa=1.0):
    assert x.ndim == 1
    x = x.reshape(-1, sdim)
    Ecoul = get_el_el_potential_energy(x)
    return Ecoul*kappa
