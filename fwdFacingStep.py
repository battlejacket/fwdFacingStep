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

from sympy import Symbol, Eq, Abs, StrictGreaterThan, StrictLessThan, And, tanh, Or, GreaterThan, LessThan, sqrt
import numpy as np
import torch

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry import Bounds, Parameterization
from modulus.sym.geometry.primitives_2d import Rectangle, Rectangle, Line
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.utils.io.vtk import var_to_polyvtk
from ansysValidator import ansysValidator
from dataConstraint import dataConstraint



@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    Re = Symbol("Re")
    xPos = Symbol("xPos")
    x, y = Symbol("x"), Symbol("y")
    Lo, Ho = Symbol("Lo"), Symbol("Ho")
    u, v = Symbol("u"), Symbol("v")
    vel = Symbol("vel")
    p, q = Symbol("p"), Symbol("q")


    # add constraints to solver
    # specify params
    D1 = 1
    L1 = 6*D1

    stepRatio = D1-0.66
    stepHeight = stepRatio*D1

    D2 = D1-stepHeight
    L2 = 12*D1

    Wo = 0.05

    Um = 1
    rho = 1
    # nu = Um*D1/Re
    nu = Symbol("nu")
    velprof = Um*2*(1-(Abs(y)/(D1/2))**2)
    # velprof2 = (4*Um*2/(D1^2))*(D1*(y)-(y)^2)

    # param_ranges = {
    # Re: (100, 1213),
    # Lo: (0.3, 0.5),
    # Ho: (0.0, stepHeight),
    # }    
    
    param_ranges = {
    Re: (1213, 1213),
    Lo: (0.5, 0.5),
    Ho: (0.05, 0.05),
    }

    pr = Parameterization(param_ranges)

    # rho = 997 #kg m-3
    # mu = 0.0008899
    # nu = mu/rho
    # Um = Re*nu/D1
    # velprof = Um*2*(1-(Abs(y)/(D1/2))**2)


    # make geometry
    pipe1 = Rectangle((-L1, -D1/2), (0, D1/2), parameterization=pr)
    pipe2 = Rectangle((-L1/2, D1/2-D2), (L2, D1/2), parameterization=pr)
    
    pipe = pipe1+pipe2

    inlet = Line((-L1, -D1/2),(-L1, D1/2), parameterization=pr)
    outlet = Line((L2, D1/2-D2),(L2, D1/2), parameterization=pr)

    integralPlane = Line((xPos, -D1/2),(xPos, D1/2), parameterization=pr)

    obstacle = Rectangle((-Lo, -D1/2),(-Lo+Wo, (-D1/2)+Ho), parameterization=pr)

    pipe -= obstacle

    # var_to_polyvtk(obstacle.sample_boundary(
    # nr_points=1000, parameterization={Re: (800,800), Lo: (0.3, 0.3), Ho: (0.2, 0.2)}), './vtp/obstacle')
    # var_to_polyvtk(pipe2.sample_boundary(
    # nr_points=1000, parameterization={Re: (800,800), Lo: (0.3, 0.3), Ho: (0.1, 0.1)}), './vtp/pipe2')
    # var_to_polyvtk(pipe.sample_boundary(
    # nr_points=1000, parameterization={Re: (800,800), Lo: (0.3, 0.3), Ho: (0.1, 0.1)}), './vtp/pipe')

    # print("geo done")


    # make annular ring domain
    domain = Domain()

    # make list of nodes to unroll graph on
    input_keys=[Key("x"), Key("y"), Key("Re_s"), Key("Ho"), Key("Lo")]
    output_keys=[Key("u"), Key("v"), Key("p")]

    ns = NavierStokes(nu=nu, rho=rho, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    
    # flow_net = FullyConnectedArch(
    #     input_keys=input_keys,
    #     output_keys=output_keys,
    # )
    flow_net = FourierNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        frequencies=("axis", [i/2 for i in range(10)]),
        frequencies_params=("axis", [i/2 for i in range(10)]),
        )


    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")
        ] + [Node.from_sympy(Re/1213, "Re_s")
        ] + [Node.from_sympy(Um*D1/Re, "nu")
        ] + [Node.from_sympy(sqrt(u**2 + v**2), "vel")
        ] + [Node.from_sympy(0.5*rho*vel**2, "q")
        ] + [Node.from_sympy(p+q, "ptot")

        ]
    )


    # # inlet
    # inletConstraint = PointwiseBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=inlet,
    #     outvar={"u": velprof, "v": 0},
    #     batch_size=cfg.batch_size.inlet,
    #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
    #     lambda_weighting={"u": 50, "v": 50},
    #     parameterization=pr,
    # )
    # domain.add_constraint(inletConstraint, "inlet")

    # # outlet
    # outletConstraint = PointwiseBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=outlet,
    #     outvar={"p": 0},
    #     batch_size=cfg.batch_size.outlet,
    #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
    #     lambda_weighting={"p": 1},
    #     parameterization=pr,
    # )
    # domain.add_constraint(outletConstraint, "outlet")

    # # # no slip
    # no_slip = PointwiseBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=pipe,
    #     outvar={"u": 0, "v": 0},
    #     batch_size=cfg.batch_size.NoSlip,
    #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
    #     lambda_weighting={"u": 1, "v": 1},
    #     criteria=And(StrictGreaterThan(x, -L1), StrictLessThan(x, L2), Or(StrictLessThan(x,-D1), StrictGreaterThan(x,D1))),
    #     parameterization=pr,
    # )
    # domain.add_constraint(no_slip, "no_slip")

    # no_slipHR = PointwiseBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=pipe,
    #     outvar={"u": 0, "v": 0},
    #     batch_size=cfg.batch_size.NoSlipHR,
    #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
    #     lambda_weighting={"u": 1, "v": 1},
    #     criteria=And(StrictGreaterThan(x, -L1), StrictLessThan(x, L2), GreaterThan(x,-D1), LessThan(x,D1)),
    #     parameterization=pr,
    # )
    # domain.add_constraint(no_slipHR, "no_slipHR")


    # # interior
    # interior = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=pipe,
    #     outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
    #     batch_size=cfg.batch_size.Interior,
    #     lambda_weighting={
    #         "continuity": 10*tanh(10 * Symbol("sdf")),
    #         "momentum_x": 10*tanh(10 * Symbol("sdf")),
    #         "momentum_y": 10*tanh(10 * Symbol("sdf")),
    #     },
    #     criteria=Or(StrictLessThan(x,-D1), StrictGreaterThan(x,D1)),
    #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
    #     # quasirandom=True,
    #     parameterization=pr,
    # )
    # domain.add_constraint(interior, "interior")

    # interiorHR = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=pipe,
    #     outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
    #     batch_size=cfg.batch_size.InteriorHR,
    #     lambda_weighting={
    #         "continuity": 10*tanh(10 * Symbol("sdf")),
    #         "momentum_x": 10*tanh(10 * Symbol("sdf")),
    #         "momentum_y": 10*tanh(10 * Symbol("sdf")),
    #     },
    #     criteria=And(GreaterThan(x,-D1), LessThan(x,D1)),
    #     batch_per_epoch=cfg.batch_size.batchPerEpoch,
    #     parameterization=pr,
    # )
    # domain.add_constraint(interiorHR, "interiorHR")

    # # integral continuity

    # def interiorCriteria(invar, params):
    #     sdf = pipe.sdf(invar, params)
    #     return np.greater(sdf["sdf"], 0)

    # integral_continuity = IntegralBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=integralPlane,
    #     # geometry=pipe,
    #     outvar={"normal_dot_vel": Um*2*(1 - (1/3)/(D1**2))},
    #     batch_size=10,
    #     integral_batch_size=cfg.batch_size.integralContinuity,
    #     lambda_weighting={"normal_dot_vel": 0.1},
    #     parameterization={**param_ranges, **{xPos:(-L1, L2)}},
    #     # fixed_dataset=False,
    #     criteria=interiorCriteria
    #     # criteria=Eq(x, xPos)
    # )
    # domain.add_constraint(integral_continuity, "integral_continuity")

    # -----------------------------------------------Data Constraints------------------------------------------------
    ansysVarNames = ("Pressure [ Pa ]", "Total Pressure [ Pa ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "X [ m ]", "Y [ m ]")
    modulusVarNames = ("p", "ptot", "u", "v", "x", "y")
    scales = ((0,1), (0,1), (0,1), (0,-1), (0,1), (-0.5,-1))
    additionalConstraints={"continuity": 0, "momentum_x": 0, "momentum_y": 0}


    file_path = "ansys/validators/export_ptot.csv"
    domain.add_constraint(dataConstraint(file_path, ansysVarNames, modulusVarNames, nodes, scales, batches=1, param=True, additionalConstraints=additionalConstraints), "full")

    # ------------------------------------------------Inferencers------------------------------------------------
    quasi = False
    crit = And(GreaterThan(x,-D1), LessThan(x,D1))
    nrPoints=50000
    para={Re: 100, Lo: 0.5, Ho: 0.2}
    interiorInferencer = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=nrPoints, parameterization=para, quasirandom=quasi, criteria=crit),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
    )
    domain.add_inferencer(interiorInferencer, "interior_" + str(para[Lo]).replace(".", ",") + "_" + str(para[Ho]).replace(".", ",") + "_" + str(para[Re]).replace(".", ","))

    para={Re: 1213, Lo: 0.5, Ho: 0.2}
    interiorInferencer = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=nrPoints, parameterization=para, quasirandom=quasi, criteria=crit),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
    )
    domain.add_inferencer(interiorInferencer, "interior_" + str(para[Lo]).replace(".", ",") + "_" + str(para[Ho]).replace(".", ",") + "_" + str(para[Re]).replace(".", ","))

    para={Re: 100, Lo: 0.3, Ho: 0.1}
    interiorInferencer = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=nrPoints, parameterization=para, quasirandom=quasi, criteria=crit),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
    )
    domain.add_inferencer(interiorInferencer, "interior_" + str(para[Lo]).replace(".", ",") + "_" + str(para[Ho]).replace(".", ",") + "_" + str(para[Re]).replace(".", ","))


    para={Re: 1213, Lo: 0.3, Ho: 0.1}
    interiorInferencer = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=nrPoints, parameterization=para, quasirandom=quasi, criteria=crit),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
    )
    domain.add_inferencer(interiorInferencer, "interior_" + str(para[Lo]).replace(".", ",") + "_" + str(para[Ho]).replace(".", ",") + "_" + str(para[Re]).replace(".", ","))

    para={Re: 100, Lo: 0.5, Ho: 0.0}
    interiorInferencer = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=nrPoints, parameterization=para, quasirandom=quasi, criteria=crit),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
    )
    domain.add_inferencer(interiorInferencer, "interior_" + str(para[Lo]).replace(".", ",") + "_" + str(para[Ho]).replace(".", ",") + "_" + str(para[Re]).replace(".", ","))

    para={Re: 1213, Lo: 0.5, Ho: 0.05}
    interiorInferencer = PointwiseInferencer(
        nodes=nodes,
        invar=pipe.sample_interior(nr_points=nrPoints, parameterization=para, quasirandom=quasi, criteria=crit),
        output_names=["u", "v", "p", "Re", "Re_s", "nu"],
    )
    domain.add_inferencer(interiorInferencer, "interior_" + str(para[Lo]).replace(".", ",") + "_" + str(para[Ho]).replace(".", ",") + "_" + str(para[Re]).replace(".", ","))

    #------------------------------------------Validators---------------------------------------------------------
    ansysVarNames = ("Pressure [ Pa ]", "Total Pressure [ Pa ]", "Velocity u [ m s^-1 ]", "Velocity v [ m s^-1 ]", "X [ m ]", "Y [ m ]")
    modulusVarNames = ("p", "ptot", "u", "v", "x", "y")
    scales = ((0,1), (0,1), (0,1), (0,-1), (0,1), (-0.5,-1))

    for root, dirs, files in os.walk(to_absolute_path("./ansys/validators")):
        for name in files:
            print(os.path.join(root, name))
            file_path = str(os.path.join(root, name))
            domain.add_validator(ansysValidator(file_path, ansysVarNames, modulusVarNames, nodes, scales, True), name)

    # Monitors
    nrPoints=1024
    inletPoints = inlet.sample_boundary(nrPoints, parameterization=pr)
    outletPoints = outlet.sample_boundary(nrPoints, parameterization=pr)
    upstreamPressurePoints_Re100    = integralPlane.sample_boundary(nrPoints, parameterization={**{Re: 100,   Lo: 0.5, Ho: 0.05}, **{xPos:-2*D1}})
    upstreamPressurePoints_Re1213   = integralPlane.sample_boundary(nrPoints, parameterization={**{Re: 1213, Lo: 0.5, Ho: 0.05}, **{xPos:-2*D1}})
    downstreamPressurePoints_Re100  = integralPlane.sample_boundary(nrPoints, parameterization={**{Re: 100,   Lo: 0.5, Ho: 0.05}, **{xPos:2*D1}})
    downstreamPressurePoints_Re1213 = integralPlane.sample_boundary(nrPoints, parameterization={**{Re: 1213, Lo: 0.5, Ho: 0.05}, **{xPos:2*D1}})


    #inlet_flow
    inlet = PointwiseMonitor(
        inletPoints,
        output_names=["normal_dot_vel", "area"],
        metrics={"inlet_flow": lambda var: torch.sum(var["area"] * var["normal_dot_vel"])},
        nodes=nodes,
    )
    domain.add_monitor(inlet)

    #outlet_flow
    outlet = PointwiseMonitor(
        outletPoints,
        output_names=["normal_dot_vel", "area"],
        metrics={"outlet_flow": lambda var: torch.sum(var["area"] * var["normal_dot_vel"])},
        nodes=nodes,
    )
    domain.add_monitor(outlet)

    #upsteramPressure
    upsteramPressure_Re100 = PointwiseMonitor(
        invar=upstreamPressurePoints_Re100,
        output_names=["p"],
        metrics={"upsteramPressure_Re100": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(upsteramPressure_Re100)

    upsteramPressure_Re1213 = PointwiseMonitor(
        invar=upstreamPressurePoints_Re1213,
        output_names=["p"],
        metrics={"upsteramPressure_Re1213": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(upsteramPressure_Re1213)

    upsteramPressureTot_Re1213 = PointwiseMonitor(
        invar=upstreamPressurePoints_Re1213,
        output_names=["ptot"],
        metrics={"upsteramPressureTot_Re1213": lambda var: torch.mean(var["ptot"])},
        nodes=nodes,
    )
    domain.add_monitor(upsteramPressureTot_Re1213)

    #downsteramPressure
    downsteramPressure_Re100 = PointwiseMonitor(
        invar=downstreamPressurePoints_Re100,
        output_names=["p"],
        metrics={"downsteramPressure_Re100": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(downsteramPressure_Re100)

    downsteramPressure_Re1213 = PointwiseMonitor(
        invar=downstreamPressurePoints_Re1213,
        output_names=["p"],
        metrics={"downsteramPressure_Re1213": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(downsteramPressure_Re1213)

    downsteramPressureTot_Re1213 = PointwiseMonitor(
        invar=downstreamPressurePoints_Re1213,
        output_names=["ptot"],
        metrics={"downsteramPressureTot_Re1213": lambda var: torch.mean(var["ptot"])},
        nodes=nodes,
    )
    domain.add_monitor(downsteramPressureTot_Re1213)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
