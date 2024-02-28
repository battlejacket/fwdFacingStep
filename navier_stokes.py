# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number

from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node


class NavierStokes_t(PDE):
    """
    Compressible Navier Stokes equations
    Reference:
    https://turbmodels.larc.nasa.gov/implementrans.html

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - 2*nu__x*u__x - nu__y*u__y - nu__y*v__x + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*u__y - nu__x*v__x - 2*nu__y*v__y + p__y
    """

    name = "NavierStokes_t"

    def __init__(self, nu, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        true_u = Symbol("true_u")
        true_v = Symbol("true_v")
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        true_p = Symbol("p")

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity_t"] = (
            rho.diff(t) + (rho * true_u).diff(x) + (rho * true_v).diff(y) + (rho * w).diff(z)
        )

        if not self.mixed_form:
            curl = Number(0) if rho.diff(x) == 0 else true_u.diff(x) + true_v.diff(y) + w.diff(z)
            self.equations["momentum_x_t"] = (
                (rho * true_u).diff(t)
                + (
                    true_u * ((rho * true_u).diff(x))
                    + true_v * ((rho * true_u).diff(y))
                    + w * ((rho * true_u).diff(z))
                    + rho * true_u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * true_u.diff(x)).diff(x)
                - (mu * true_u.diff(y)).diff(y)
                - (mu * true_u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
                - mu.diff(x) * true_u.diff(x)
                - mu.diff(y) * true_v.diff(x)
                - mu.diff(z) * w.diff(x)
            )
            self.equations["momentum_y_t"] = (
                (rho * true_v).diff(t)
                + (
                    true_u * ((rho * true_v).diff(x))
                    + true_v * ((rho * true_v).diff(y))
                    + w * ((rho * true_v).diff(z))
                    + rho * true_v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * true_v.diff(x)).diff(x)
                - (mu * true_v.diff(y)).diff(y)
                - (mu * true_v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
                - mu.diff(x) * true_u.diff(y)
                - mu.diff(y) * true_v.diff(y)
                - mu.diff(z) * w.diff(y)
            )

