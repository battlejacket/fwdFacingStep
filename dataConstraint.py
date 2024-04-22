from modulus.sym.hydra import to_absolute_path
from sympy import Symbol
from os import path
from csv_rw import csv_to_dict
import numpy as np
from modulus.sym.domain.constraint import PointwiseConstraint
from sympy import Symbol, Eq, Abs, StrictGreaterThan, StrictLessThan, And, tanh, Or, GreaterThan, LessThan, Not, sqrt, lambdify, Heaviside


Re = Symbol("Re")
x, y = Symbol("x"), Symbol("y")
Lo, Ho = Symbol("Lo"), Symbol("Ho")

def dataConstraint(file_path, ansysVarNames, modulusVarNames, nodes, scales, batches, skiprows=1, param=False, nonDim=None, additionalConstraints=None, geo=None, lambdaFn=NotImplemented):
    if path.exists(to_absolute_path(file_path)):
        mapping = {}
        for ansVarName, modulusVarName in zip(ansysVarNames, modulusVarNames):
            mapping[ansVarName] = modulusVarName

        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping, skiprows=skiprows)

        if param!=False:
            parameterRanges = param               
        
            openfoam_var.update({"Re": np.full_like(openfoam_var["x"], parameterRanges[Re])})
            openfoam_var.update({"Lo": np.full_like(openfoam_var["x"], parameterRanges[Lo])})
            openfoam_var.update({"Ho": np.full_like(openfoam_var["x"], parameterRanges[Ho])})
        
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
        
        

            
        lambda_weighting={"u_d": np.full_like(openfoam_outvar_numpy["u_d"], 0.1), "v_d": np.full_like(openfoam_outvar_numpy["v_d"], 0.1), "p_d": np.full_like(openfoam_outvar_numpy["p_d"], 0.1)}
        
        
        if geo!=None:
            sdfInvarKeys = ["x", "y"]
            sdfParamKeys = ["Re", "Lo", "Ho"]
        
            sdfInvar = {
                key: value
                for key, value in openfoam_var.items()
                if key in sdfInvarKeys
            }
            
            sdfParam = {
                key: value
                for key, value in openfoam_var.items()
                if key in sdfParamKeys
            }
            
            sdf = geo.sdf(sdfInvar, sdfParam)
            # lambdaFn = 1*tanh(20 * var)
            
            lambdaFnlf = lambdify(Symbol("sdf"), lambdaFn, "numpy")
            
            lambdaNumpy = lambdaFnlf(sdf['sdf'])
            
            for key in additionalConstraints.keys():
                lambda_weighting[key] = lambdaNumpy
            
        
        openfoam_invar_numpy_t={}
        openfoam_outvar_numpy_t={}
        
        
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
            lambda_weighting=lambda_weighting
            )
        return dataConstraint
    else:
        print("Missing Data: ", file_path)