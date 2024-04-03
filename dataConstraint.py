from modulus.sym.hydra import to_absolute_path
from sympy import Symbol
from os import path
from csv_rw import csv_to_dict
import numpy as np
from modulus.sym.domain.constraint import PointwiseConstraint



Re = Symbol("Re")
x, y = Symbol("x"), Symbol("y")
Lo, Ho = Symbol("Lo"), Symbol("Ho")

def dataConstraint(file_path, ansysVarNames, modulusVarNames, nodes, scales, batches, skiprows=1, param=False, nonDim=None, additionalConstraints=None, criteria=None):
    if path.exists(to_absolute_path(file_path)):
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
        
        if additionalConstraints != None:
            for key, value in additionalConstraints.items():
                openfoam_var.update({key: np.full_like(openfoam_var["x"], value)})
                outvarKeys += (key,) 

        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in invarKeys

        }

        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in outvarKeys
        }
        
        
        
        openfoam_invar_numpy_t={}
        openfoam_outvar_numpy_t={}
        
        # for critKey in criteria.keys():
        
        # for key in openfoam_invar_numpy.keys():
        #     openfoam_invar_numpy_t[key] = np.copy(openfoam_invar_numpy[key][(openfoam_invar_numpy["y"]<-0.147) & (openfoam_invar_numpy["y"]>-0.187)])
        #     openfoam_invar_numpy_t[key] = openfoam_invar_numpy_t[key].reshape((openfoam_invar_numpy_t[key].shape[0], 1))
        # for key in openfoam_outvar_numpy.keys():
        #     openfoam_outvar_numpy_t[key] = np.copy(openfoam_outvar_numpy[key][(openfoam_invar_numpy["y"]<-0.147) & (openfoam_invar_numpy["y"]>-0.187)])
        #     openfoam_outvar_numpy_t[key]=openfoam_outvar_numpy_t[key].reshape((openfoam_outvar_numpy_t[key].shape[0], 1))
        # openfoam_invar_numpy=openfoam_invar_numpy_t
        # openfoam_outvar_numpy=openfoam_outvar_numpy_t
        

        # print(openfoam_var['x'].size)
        dataConstraint = PointwiseConstraint.from_numpy(
            nodes=nodes, 
            invar=openfoam_invar_numpy, 
            outvar=openfoam_outvar_numpy, 
            batch_size=int(openfoam_invar_numpy['x'].size/batches),
            lambda_weighting={"u_d": np.full_like(openfoam_outvar_numpy["u_d"], 0.1), "v_d": np.full_like(openfoam_outvar_numpy["v_d"], 0.1), "p_d": np.full_like(openfoam_outvar_numpy["p_d"], 0.1)}
            )
        return dataConstraint
    else:
        print("Missing Data: ", file_path)