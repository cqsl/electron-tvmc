from typing import Optional, Callable, Union
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import netket as nk
from netket.utils.types import DType, PyTree, Array
import netket.jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray


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


def jacfwd(f, vec=None):
    def jacfun(x):
        jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
        if vec is None:
            eye = jnp.eye(len(x), dtype=x.dtype)
            J = jax.vmap(jvp_fun, in_axes=0)(eye)
        else:
            J = jvp_fun(vec)
        return J

    return jacfun


class KineticEnergy(ContinuousOperator):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: Union[float, list[float]],
        dtype: Optional[DType] = None,
        mode: str = "parallel",
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        mass: float if all masses are the same, list indicating the mass of each particle otherwise
        dtype: Data type of the matrix elements. Defaults to `np.float64`
        mode: Method to compute the kinetic energy: "sequential", "parallel", "jet", "folx" and "fwdlap"
                (sequential uses less memory compared to parallel)
        """

        self._mass = jnp.asarray(mass, dtype=dtype)

        self._is_hermitian = np.allclose(self._mass.imag, 0.0)
        self.__attrs = None

        self._mode = mode.strip().lower()

        super().__init__(hilbert, self._mass.dtype)

    @property
    def mass(self):
        return self._mass

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel_single(
        self, logpsi: Callable, params: PyTree, x: Array, inverse_mass: Optional[PyTree]
    ):
        def logpsi_x(x):
            return logpsi(params, x)

        # multiple methods to compute the second order derivatives
        if self._mode == "parallel":
            """Uses the standard netket approach"""
            dlogpsi_x = jacrev(logpsi_x)
            dp_dx = dlogpsi_x(x)[0][0] ** 2
            dp_dx2 = jnp.diag(jacfwd(dlogpsi_x)(x)[0].reshape(x.shape[0], x.shape[0]))
            return -0.5 * jnp.sum(inverse_mass * (dp_dx2 + dp_dx), axis=-1)

        elif self._mode == "sequential":
            """Sequentially computes the kinetic terms in FermiNet style."""
            dlogpsi_x = jacrev(logpsi_x)
            dp_dx = dlogpsi_x(x)[0][0] ** 2

            def make_hess(carry, vec):
                value = jacfwd(dlogpsi_x, vec=vec)(x)[0]
                return carry, value

            _, dp_dx2 = jax.lax.scan(
                make_hess, None, jnp.eye(x.shape[0], dtype=x.dtype)
            )
            dp_dx2 = jnp.diag(dp_dx2.reshape(x.shape[0], x.shape[0]))
            return -0.5 * jnp.sum(inverse_mass * (dp_dx2 + dp_dx), axis=-1)

        elif self._mode == "jet":
            """Using jet higher order derivatives."""
            from jax.experimental import jet

            def calc_ke_jet(f, x):
                @jax.vmap
                def hvv(v):
                    _, (df, dff) = jet.jet(f, (x,), ((v, jnp.zeros_like(x)),))
                    return df, dff

                basis_vec = jnp.eye(x.shape[0], dtype=x.dtype)
                dp_dx, dp_dx2 = hvv(basis_vec)
                return -0.5 * inverse_mass * (jnp.sum(dp_dx2) + jnp.sum(dp_dx**2))

            return calc_ke_jet(logpsi_x, x)

        elif self._mode == "folx":
            """Uses: https://github.com/microsoft/folx """
            ## Add a WARNING! -- only works with real wavefunctions (though the `jvp.py` and `hessian.py`
            ## source files can be modified to make it work with complex wavefunctions)
            from folx import forward_laplacian

            lap_log_pdf = forward_laplacian(logpsi_x)
            res = lap_log_pdf(x)
            dp_dx = jnp.squeeze(res.jacobian.dense_array) ** 2
            dp_dx2 = res.laplacian
            folx_kin = -0.5 * inverse_mass * dp_dx2 - 0.5 * inverse_mass * jnp.sum(
                dp_dx
            )
            return folx_kin

        elif self._mode == "fwdlap":
            """Uses: https://github.com/y1xiaoc/fwdlap
            pip install git+https://github.com/y1xiaoc/fwdlap
            """
            import fwdlap

            def calc_ke_fwdlap(log_psi, x, inner_size=None, batch_size=None):
                def _lapl_over_psi(x):
                    x_shape = x.shape
                    flat_x = x.reshape(-1)
                    ncoord = flat_x.size
                    f = lambda flat_x: log_psi(
                        flat_x.reshape(x_shape)
                    )  # take flattened x
                    eye = jnp.eye(ncoord, dtype=x.dtype)
                    zero = fwdlap.Zero.from_value(flat_x)
                    primals, dp_dx, dp_dx2 = fwdlap.lap(f, (flat_x,), (eye,), (zero,))
                    laplacian = (
                        inverse_mass * (dp_dx**2).sum() + inverse_mass * dp_dx2
                    )
                    return laplacian

                return -0.5 * _lapl_over_psi(x)

            return calc_ke_fwdlap(logpsi_x, x)

        else:
            raise ValueError(
                f"method {self._mode} to compute the kinetic energy is unknown, the options are parallel or sequential."
            )

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, coefficient: Optional[PyTree]
    ):
        return self._expect_kernel_single(logpsi, params, x, coefficient)

    def _pack_arguments(self) -> PyTree:
        return 1.0 / self._mass

    @property
    def _attrs(self):
        if self.__attrs is None:
            self.__attrs = (self.hilbert, self.dtype, HashableArray(self.mass))
        return self.__attrs

    def __repr__(self):
        return f"KineticEnergy(m={self._mass}, mode={self._mode})"


class PotentialEnergyTD(nk.operator.PotentialEnergy):
    r"""Returns the local potential energy defined in afun(x, t)"""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        coefficient: float = 1.0,
        t: float = 0.0,
        dtype: Optional[DType] = float,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            afun: The potential energy as function of x
            coefficients: A coefficient for the ContinuousOperator object
            dtype: Data type of the matrix elements. Defaults to `np.float64`
        """
        super().__init__(hilbert, afun, coefficient=coefficient, dtype=dtype)
        self._t = jnp.array(t, dtype=dtype)

    def set_t(self, t):
        self.t = t
        return self

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, _t):
        self._t = jnp.array(_t, dtype=self.dtype)

    def _expect_kernel_single(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        coefficient, t = data
        return coefficient * self._afun(x, t=t)

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        return self._expect_kernel_single(logpsi, params, x, data)

    @property
    def is_hermitian(self):
        return True

    def _pack_arguments(self):
        return (self.coefficient, self.t)

    def __repr__(self):
        return f"PotentialEnergyTD(t={self.t}, coefficient={self.coefficient}, function={self._afun})"
