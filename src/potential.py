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