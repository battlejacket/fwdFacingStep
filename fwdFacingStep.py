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

import os
import warnings
import numpy as np

from sympy import Symbol, Eq, Abs, StrictGreaterThan, StrictLessThan, And, srepr

import modulus.sym
from modulus.sym import quantity
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Channel2D, Line, Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.node import Node
from modulus.sym.geometry import Parameterization
from modulus.sym.eq.non_dim import NonDimensionalizer
from modulus.sym import quantity
from modulus.sym.eq.pdes.basic import NormalDotVec


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    Re = Symbol("Re")
    x, y = Symbol("x"), Symbol("y")
    x_s, y_s = Symbol("x"), Symbol("y")
    u, v, p = Symbol("u"), Symbol("v"), Symbol("p")
    u_s, v_s, p_s = Symbol("u_s"), Symbol("v_s"), Symbol("p_s")
    
    # make geometry
    D1 = 1
    L1 = 3*D1

    ratio = 1.87

    D2 = D1/ratio
    L2 = 6*D1

    param_ranges = {
    Re: (800, 1213)
    }

    pr = Parameterization(param_ranges)

    # rho = 997 #kg m-3
    # mu = 0.0008899
    # nu = mu/rho
    # Um = Re*nu/D1
    # velprof = Um*2*(1-(Abs(y)/(D1/2))**2)

    Um = 1
    rho = 1
    # nu = Um*D1/Re
    nu = Symbol("nu")
    velprof = Um*2*(1-(Abs(y)/(D1/2))**2)

    pipe1 = Rectangle((-L1, -D1/2), (0, D1/2))
    pipe2 = Rectangle((0, -D2/2), (L2, D2/2))
    
    pipe = pipe1+pipe2

    inlet = Line((-L1, -D1/2),(-L1, D1/2))
    outlet = Line((L2, -D2/2),(L2, D2/2))

    # var_to_polyvtk(pipe1.sample_boundary(
    # nr_points=1000, ), './vtp/pipe1')
    
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu, rho=rho, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("Re_s")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")
                            #    ] + [Node.from_sympy(u_s*Um, "u")
                            #    ] + [Node.from_sympy(v_s*Um, "v")
                            #    ] + [Node.from_sympy(p_s*1, "p")
                               ] + [Node.from_sympy(Re/1213, "Re_s")
                               ] + [Node.from_sympy(Um*D1/Re, "nu")
                            #    ] + [Node.from_sympy(x/L2, "x_s")
                            #    ] + [Node.from_sympy(y/L2, "y_s")
                                    ]


    # make ldc domain
    ldc_domain = Domain()

    # inlet
    inletConstraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": velprof, "v": 0},
        batch_size=cfg.batch_size.inlet,
        batch_per_epoch=4000,
        lambda_weighting={"u": 1, "v": 1},
        parameterization=pr,
    )
    ldc_domain.add_constraint(inletConstraint, "inlet")

    outletConstraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        batch_per_epoch=4000,
        lambda_weighting={"p": 1},
        parameterization=pr,
    )
    ldc_domain.add_constraint(outletConstraint, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=pipe,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        batch_per_epoch=4000,
        lambda_weighting={"u": 1, "v": 1},
        criteria=And(StrictGreaterThan(x, -L1), StrictLessThan(x, L2)),
        parameterization=pr,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=pipe,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            # "continuity": (20)*Symbol("sdf"),
            # "momentum_x": (20)*Symbol("sdf"),
            # "momentum_y": (20)*Symbol("sdf"),
            "continuity": (10)*(1-Abs(y)),
            "momentum_x": (10)*(1-Abs(y)),
            "momentum_y": (10)*(1-Abs(y)),
        },
        parameterization=pr,
    )
    ldc_domain.add_constraint(interior, "interior")

    # # integral continuity
    # integral_continuity = IntegralBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=geo,
    #     outvar={"normal_dot_vel": 2},
    #     batch_size=10,
    #     integral_batch_size=cfg.batch_size.integral_continuity,
    #     lambda_weighting={"normal_dot_vel": 0.1},
    #     criteria=Eq(x, channel_length[1]),
    # )
    # domain.add_constraint(integral_continuity, "integral_continuity")

    interiorInferencer1 = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=10000,parameterization={Re: 800}),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
        # output_names=["u", "v", "p", "u_s", "v_s", "p_s", "x_s", "y_s", "Re_s"],
        # batch_size=1024,
        # plotter=InferencerPlotter(),
    )
    ldc_domain.add_inferencer(interiorInferencer1, "interior_800")

    interiorInferencer2 = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=10000,parameterization={Re: 1213}),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
        # output_names=["u", "v", "p", "u_s", "v_s", "p_s", "x_s", "y_s", "Re_s"],
        # batch_size=1024,
        # plotter=InferencerPlotter(),
    )
    ldc_domain.add_inferencer(interiorInferencer2, "interior_1213")


    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
