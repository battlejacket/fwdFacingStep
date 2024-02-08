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

from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym import quantity
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Channel2D, Line
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
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


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # make geometry
    D1_nd = 1 #quantity(60, "mm")
    L1_nd = 50*D1_nd
    A1_nd = np.pi*(D1_nd/2)**2

    D2_nd = 0.5 #quantity(15, "mm")
    L2_nd = 50*D2_nd
    A2_nd = np.pi*(D2_nd/2)**2

    x, y = Symbol("x"), Symbol("y")

    pipe1 = Channel2D((-L1_nd, -D1_nd/2), (0, D1_nd/2))
    pipe2 = Channel2D((0, -D2_nd/2), (L2_nd, D2_nd/2))
    step1 = Line((0, -D1_nd/2), (0, -D2_nd/2))
    step2 = Line((0, D2_nd/2), (0, D1_nd/2))
    
    
    pipe = pipe1+step1+step2

    inlet = Line((-L1_nd, -D1_nd/2),(-L1_nd, D1_nd/2))
    outlet = Line((L2_nd, -D2_nd/2),(L2_nd, D2_nd/2))

    var_to_polyvtk(pipe.sample_boundary(
    nr_points=1000, ), './vtp/pipe')


    # make ldc domain
    ldc_domain = Domain()

    # inlet
    inletConstraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": 1.0, "v": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={"u": 1, "v": 1.0},  # weight edges to be zero
    )
    ldc_domain.add_constraint(inletConstraint, "inlet")

    outletConstraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={"p": 1},  # weight edges to be zero
    )
    ldc_domain.add_constraint(outletConstraint, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=pipe,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=pipe,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": 20*Symbol("sdf"),
            "momentum_x": 20*Symbol("sdf"),
            "momentum_y": 20*Symbol("sdf"),
        },
    )
    ldc_domain.add_constraint(interior, "interior")

    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
