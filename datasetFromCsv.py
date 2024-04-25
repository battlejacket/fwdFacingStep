from modulus.sym.hydra import to_absolute_path
from os import path
from csv_rw import csv_to_dict
import numpy as np
from modulus.sym.domain.constraint import PointwiseConstraint
from sympy import Symbol, lambdify
from modulus.sym.geometry import Parameterization

from modulus.sym.geometry.helper import (
    _concat_numpy_dict_list,
    _sympy_criteria_to_criteria,
    )
Re = Symbol("Re")
x, y = Symbol("x"), Symbol("y")
Lo, Ho = Symbol("Lo"), Symbol("Ho")

def readCsvDataToDict(filePath, csvInvarNames, csvOutvarNames, modulusInvarNames, modulusOutvarNames, scales, skiprows, criteria):
    if path.exists(to_absolute_path(filePath)):
        csvVarNames = csvInvarNames + csvOutvarNames
        modulusVarNames = modulusInvarNames + modulusOutvarNames
        
        mapping = {}
        for csvVarName, modulusVarName in zip(csvVarNames, modulusVarNames):
            mapping[csvVarName] = modulusVarName

        csvData = csv_to_dict(to_absolute_path(filePath), mapping, skiprows=skiprows)
        
        for key in modulusVarNames:
            csvData[key] += scales[key][0]
            csvData[key] /= scales[key][1]

        csvInvar = {
            key: value for key, value in csvData.items() if key in modulusInvarNames
        }

        csvOutvar = {
            key: value for key, value in csvData.items() if key in modulusOutvarNames
        }
        
        if criteria!=None:
            criteria = _sympy_criteria_to_criteria(criteria)
    
            criteriaNumpy = criteria(csvInvar, {})
            
            for key in csvInvar.keys():
                csvInvar[key]=csvInvar[key][criteriaNumpy>0]
                csvInvar[key] = csvInvar[key].reshape((csvInvar[key].shape[0], 1))
            
            for key in csvOutvar.keys():
                csvOutvar[key]=csvOutvar[key][criteriaNumpy>0]
                csvOutvar[key] = csvOutvar[key].reshape((csvOutvar[key].shape[0], 1))
        
        return csvInvar, csvOutvar
    else:
        print("Missing Data: ", filePath)

def datasetFromCsv(filePath, csvInvarNames, csvOutvarNames, modulusInvarNames, modulusOutvarNames, scales, skiprows=1, parameterRanges=None, criteria=None, additionalConstraints=None, geo=None, lambdaWeighting=None):

    dataInvar, dataOutvar = readCsvDataToDict(filePath, csvInvarNames, csvOutvarNames, modulusInvarNames, modulusOutvarNames, scales, skiprows, criteria)

    if parameterRanges!=None:
        dataParam={}            
        for key in parameterRanges.keys():
            dataParam[str(key)]= np.full_like(dataInvar["x"], parameterRanges[key])
        dataInvar={**dataInvar, **dataParam}
    
    if additionalConstraints != None:
        for key, value in additionalConstraints.items():
            dataOutvar.update({str(key): np.full_like(dataOutvar["x"], value)})
            # modulusOutvarNames += (str(key),) 
    
    if lambdaWeighting!=None:
        lambdaWeights={}
        for key, value in lambdaWeighting.items():
            lambdaWeights[key] = np.full_like(dataOutvar[key], value)
        return dataInvar, dataOutvar, lambdaWeights
    else:
        return dataInvar, dataOutvar
        
    # if geo!=None:
    #     sdf = geo.sdf(splitInvar, splitParam)
    #     # lambdaFn = 1*tanh(20 * var)
        
    #     lambdaFnlf = lambdify(Symbol("sdf"), lambdaFn, "numpy")
        
    #     lambdaNumpy = lambdaFnlf(sdf['sdf'])
        
    #     for key in additionalConstraints.keys():
    #         lambda_weighting[key] = lambdaNumpy
    

    # print(data_var['x'].size)

    # return {**dataInvar, **dataParam}, dataOutvar, lambdaWeights