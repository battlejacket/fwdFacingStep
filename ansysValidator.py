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


def ansysValidator(file_path, ansysVarNames, modulusVarNames, nodes, scales, param=False, nonDim=None):
                if os.path.exists(to_absolute_path(file_path)):
                    mapping = {}
                    for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
                        mapping[ansVarName] = modulusVarName

                    openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping, skiprows=6)

                    if param:
                        # parameters = file_path.split("_")[2].split(".")[0].replace(",", ".").split("-")
                        parameterRanges = {
                            Re: 1213,  
                            Lo: 0.5,
                            Ho: 0.05,
                        }                  
                    
                        openfoam_var.update({"Re": np.full_like(openfoam_var["x"], parameterRanges[Re])})
                        openfoam_var.update({"Lo": np.full_like(openfoam_var["x"], parameterRanges[Lo])})
                        openfoam_var.update({"Ho": np.full_like(openfoam_var["x"], parameterRanges[Ho])})
                    
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

                    np.save(file="inv", arr=openfoam_invar_numpy)

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

            # def ansysValidator(file_path, ansysVarNames, modulusVarNames, parameterRanges):
            #     if os.path.exists(to_absolute_path(file_path)):
            #         mapping = {}
            #         for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
            #             mapping[ansVarName] = modulusVarName

            #         openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
            #         openfoam_var["x"] /= length_scale_ch.magnitude
            #         openfoam_var["y"] /= length_scale_ch.magnitude
            #         openfoam_var["z"] /= length_scale_ch.magnitude
                    
            #         openfoam_var.update({"holeRadius1": np.full_like(openfoam_var["x"], parameterRanges[holeRadius1])})
            #         openfoam_var.update({"holeRadius2": np.full_like(openfoam_var["x"], parameterRanges[holeRadius2])})
            #         openfoam_var.update({"inletVelocity1": np.full_like(openfoam_var["x"], parameterRanges[inletVelocity1])})
            #         openfoam_var.update({"inletVelocity2": np.full_like(openfoam_var["x"], parameterRanges[inletVelocity2])})
            #         openfoam_var.update({"holePosX1": np.full_like(openfoam_var["x"], parameterRanges[holePosX1])})
            #         openfoam_var.update({"holePosX2": np.full_like(openfoam_var["x"], parameterRanges[holePosX2])})
            #         openfoam_var.update({"holePosZ1": np.full_like(openfoam_var["x"], parameterRanges[holePosZ1])})
            #         openfoam_var.update({"holePosZ2": np.full_like(openfoam_var["x"], parameterRanges[holePosZ2])})

            #         openfoam_invar_numpy = {
            #             key: value
            #             for key, value in openfoam_var.items()
            #             if key in ["x", "y", "z", "holeRadius1", "inletVelocity1", "holeRadius2", "inletVelocity2", "holePosX1", "holePosX2", "holePosZ1", "holePosZ2"]
            #         }    
                            
            #         openfoam_outvar_numpy = {
            #             key: value for key, value in openfoam_var.items() if key in modulusVarNames[:-3]
            #         }

            #         openfoam_validator = PointwiseValidator(
            #             nodes=thermal_nodes,
            #             invar=openfoam_invar_numpy,
            #             true_outvar=openfoam_outvar_numpy,
            #             batch_size=3000,
            #             # plotter=CustomValidatorPlotter(),
            #             # requires_grad=True,
            #         )
            #         return openfoam_validator