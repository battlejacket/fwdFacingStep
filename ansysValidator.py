from modulus.sym.hydra import to_absolute_path
from sympy import Symbol
import os
from parameterRangeContainer import parameterRangeContainer
from csv_rw import csv_to_dict
import numpy as np
from modulus.sym.domain.validator import PointwiseValidator



Re = Symbol("Re")
x, y = Symbol("x"), Symbol("y")
Lo, Ho = Symbol("Lo"), Symbol("Ho")


def ansysValidator(file_path, ansysVarNames, modulusVarNames, nodes, scales, skiprows=1, param=False, nonDim=None):
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {}
        for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
            mapping[ansVarName] = modulusVarName

        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping, skiprows=skiprows)

        if param:
            parameters = file_path.split("_")[1].split(".")[0].replace(",", ".").split("-")

            parameterRanges = {
                Re: float(parameters[0]),  
                Lo: float(parameters[1]),
                Ho: float(parameters[2]),
            } 


            # parameterRanges = {
            #     Re: 1213,
            #     Lo: 0.5,
            #     Ho: 0.05,
            # }                  
        
            openfoam_var.update({"Re": np.full_like(openfoam_var["x"], parameterRanges[Re])})
            openfoam_var.update({"Lo": np.full_like(openfoam_var["x"], parameterRanges[Lo])})
            openfoam_var.update({"Ho": np.full_like(openfoam_var["x"], parameterRanges[Ho])})
            # openfoam_var.update({"continuity": np.full_like(openfoam_var["x"], 0)})
            # openfoam_var.update({"momentum_x": np.full_like(openfoam_var["x"], 0)})
            # openfoam_var.update({"momentum_y": np.full_like(openfoam_var["x"], 0)}
        
        for key, scale in zip(modulusVarNames, scales):
            openfoam_var[key] += scale[0]
            openfoam_var[key] /= scale[1]
            

        invarKeys = ["x", "y", "Re", "Lo", "Ho"]
        outvarKeys = modulusVarNames[:-2]
        

        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in invarKeys

        }

        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in outvarKeys
        }

        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=openfoam_invar_numpy['x'].size,
            # plotter=CustomValidatorPlotter(),
            requires_grad=True,
        )
        return openfoam_validator
    else:
        print("Missing Data: ", file_path)
