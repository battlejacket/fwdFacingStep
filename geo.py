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

# import os
from os import walk, path
import csv

from sympy import Symbol, Eq, Abs, StrictGreaterThan, StrictLessThan, And, tanh, Or, GreaterThan, LessThan, Not, sqrt, Heaviside
import numpy as np
import torch

import modulus.sym
from modulus.sym.hydra import to_absolute_path, ModulusConfig
# from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_2d import Rectangle, Rectangle, Line
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)

from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.models.fourier_net import FourierNetArch
from modulus.sym.utils.io.vtk import var_to_polyvtk
from ansysValidator import ansysValidator
from dataConstraint import dataConstraint
from readParameters import readParametersFromFileName, readParametersFromCSV
from datasetFromCsv import datasetFromCsv

Re = Symbol("Re")
xPos = Symbol("xPos")
x, y = Symbol("x"), Symbol("y")
Lo, Ho = Symbol("Lo"), Symbol("Ho")
u, v, u_norm = Symbol("u"), Symbol("v"), Symbol("u_norm")
vel = Symbol("vel")
p, q, p_norm= Symbol("p"), Symbol("q"), Symbol("p_norm")
nu = Symbol("nu")


# add constraints to solver
# specify params
D1 = 1
L1 = 6*D1

# stepRatio = D1-0.66
# stepHeight = stepRatio*D1

stepHeight = 0.5*D1

D2 = D1-stepHeight
L2 = 12*D1

Wo = 0.1

Um = 1
rho = 1

velprof = Um*2*(1-(Abs(y)/(D1/2))**2)
# velprof2 = (4*Um*2/(D1^2))*(D1*(y)-(y)^2)

param_ranges = {
    Re: (100, 1000),
    Lo: (0.2, 1),
    Ho: (0.1, 0.5),
    }

# param_ranges = {
#     Re: (100, 1000),
#     Lo: (0.1, 1),
#     Ho: (0.165, 0.33),
#     } 

# param_ranges = {
#     Re: 500,
#     Lo: 0.5,
#     Ho: 0.2,
#     }


pr = Parameterization(param_ranges)

# make geometry
upstreamChannel = Rectangle((-L1, -D1/2), (0, D1/2), parameterization=pr)
downstreamChannel = Rectangle((-L1/2, -D1/2), (L2, -D1/2+D2), parameterization=pr)

channel = upstreamChannel+downstreamChannel

inlet = Line((-L1, -D1/2),(-L1, D1/2), parameterization=pr)
outlet = Line((L2, -D1/2),(L2, -D1/2+D2), parameterization=pr)

integralPlane = Line((xPos, -D1/2),(xPos, D1/2), parameterization=pr)

obstacle = Rectangle((-Lo, D1/2-Ho),(-Lo+Wo, D1/2), parameterization=pr)

channel -= obstacle

noSlipCriteria=And(StrictGreaterThan(x, -L1), StrictLessThan(x, L2)) # ignore inlet/outlet
noSlipHrCriteria=And(GreaterThan(x,-1.5*D1), LessThan(x,D1/2), GreaterThan(y, 0))
criteriaHR=And(GreaterThan(x,2*-D1), LessThan(x,2*D1))

def interiorCriteria(invar, params):
    sdf = channel.sdf(invar, params)
    return np.greater(sdf["sdf"], 0)

pointMultip=1

var_to_polyvtk(channel.sample_boundary(
nr_points=pointMultip*1024, parameterization={Re: 800, Lo: 0.3, Ho: 0.4}, criteria=And(noSlipCriteria, Not(noSlipHrCriteria))), './vtp/noSlip')

var_to_polyvtk(channel.sample_boundary(
nr_points=pointMultip*512, parameterization={Re: 800, Lo: 0.3, Ho: 0.4}, criteria=And(noSlipCriteria, noSlipHrCriteria)), './vtp/noSlipHR')

var_to_polyvtk(inlet.sample_boundary(
nr_points=pointMultip*64, parameterization={Re: 800, Lo: 0.3, Ho: 0.4}), './vtp/inlet')

var_to_polyvtk(outlet.sample_boundary(
nr_points=pointMultip*64, parameterization={Re: 800, Lo: 0.3, Ho: 0.4}), './vtp/outlet')

var_to_polyvtk(channel.sample_interior(
nr_points=pointMultip*2048, parameterization={Re: 800, Lo: 0.3, Ho: 0.4}, criteria=Not(criteriaHR)), './vtp/interior')

var_to_polyvtk(channel.sample_interior(
nr_points=pointMultip*2048, parameterization={Re: 800, Lo: 0.3, Ho: 0.4}, criteria=criteriaHR), './vtp/interiorHR')

var_to_polyvtk(integralPlane.sample_boundary(
nr_points=pointMultip*512, parameterization={Re: 800, Lo: 0.3, Ho: 0.4, xPos: -2.5}, criteria=interiorCriteria), './vtp/integralUS')

var_to_polyvtk(integralPlane.sample_boundary(
nr_points=pointMultip*512, parameterization={Re: 800, Lo: 0.3, Ho: 0.4, xPos: -0.25}, criteria=interiorCriteria), './vtp/integralOBS')

var_to_polyvtk(integralPlane.sample_boundary(
nr_points=pointMultip*512, parameterization={Re: 800, Lo: 0.3, Ho: 0.4, xPos: 2.5}, criteria=interiorCriteria), './vtp/integralDS')


print("geo done")