import numpy as np
import dill
import os, glob, io, time
from os import listdir
import csv
from fwdFacingStep import ffs #, param_ranges, Re, Ho, Lo
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.visualization.scatter import Scatter
import contextlib
from multiprocessing import Process
from pymoo.termination.default import DefaultMultiObjectiveTermination


def readFile(fileDir, objective, design):
    file = objective + "_design_" + str(design[0]) + ".csv"
    with open(os.path.join(fileDir, file), "r") as datafile:
        data = []
        reader = csv.reader(datafile, delimiter=",")
        for row in reader:
            columns = [row[1]]
            data.append(columns)
        last_row = float(data[-1][0])
        return np.array(last_row)

def evaluate(designs, path, reynoldsNr) :
    
    configFileDir = path+"/conf/"
    path_monitors = os.path.join(path, "monitors")
    
    tfFiles = glob.glob(os.path.join(path, "events.out.tfevents*"))

    returnVal = []
    # run modulus
    # with contextlib.redirect_stdout(io.StringIO()):
    p = Process(target=ffs, args=(designs,reynoldsNr, configFileDir[2:], "config", True))
    p.start()
    p.join() 
    # read result files
    for design in enumerate(designs):
        # read upstream pressure
        objective = "upstreamPressure"
        USP = readFile(fileDir = path_monitors, objective = objective, design = design)
        # read downstream pressure
        objective = "downstreamPressure"
        DSP = readFile(fileDir = path_monitors, objective = objective, design = design)
        returnVal.append(2*(USP-DSP))
            

    # remove old files
    filePattern = "*.csv"
    filePaths = glob.glob(os.path.join(path_monitors, filePattern))
    for file_path in filePaths:
        if "_design_" in file_path:
            os.remove(file_path)
    
    filePattern = "events.out.tfevents*"
    filePaths = glob.glob(os.path.join(path, filePattern))
    for file_path in filePaths:
        if file_path not in tfFiles:
            os.remove(file_path)

    return np.array(returnVal)
    
    
outputsPath="./outputs/fwdFacingStep_fl/"
dirSkip = [".hydra", "init", "initFC"]

resultsDir = "./responseResults/"

Lo = np.arange(0.2, 1.05, 0.0125)
# Lo = np.arange(0.1, 0.5, 0.0125)

# models = ["data1800PlusPhysicsLambda01@500k"]
models = listdir(outputsPath)
models.sort()

# models = ["physicsOnlyFC@500k", "physicsOnly@500k", "dataOnly1800FC@500k", "dataOnly1800@500k"]
# models = ["data1800PlusPhysicsLambda1FC@500k", "data1800PlusPhysicsLambda1@500k", "data1800PlusPhysicsLambda01FC@500k", "data1800PlusPhysicsLambda01@500k"]


for model in models:
    if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
    # if model in dirSkip or "100k" in model.split("@")[-1]:
        print("skipping ", model)
        continue
        
    path = outputsPath + model
    resultsPath = resultsDir + model
    
    for reNr in [500, 800]:
        for HoV in [0.35, 0.4, 0.45]:
        # for HoV in [0.2, 0.3, 0.4]:
            
            Ho = np.full_like(Lo, HoV)

            designs = np.array([
                [val[0], val[1]] for val in zip(Lo, Ho)
                # [val[0], val[1]] for val in zip(Ho, Lo)
                ])

            results = evaluate(designs, path, reNr)


            if not os.path.exists(resultsPath):
                os.mkdir(resultsPath)

            np.save(file=resultsPath + "/designsRe" + str(reNr) + "Ho" + str(HoV), arr=designs)
            np.save(file=resultsPath + "/resultsRe" + str(reNr) + "Ho" + str(HoV), arr=results)
            
            # np.save(file=resultsPath + "/designsRe" + str(reNr) + "Lo" + str(HoV), arr=designs)
            # np.save(file=resultsPath + "/resultsRe" + str(reNr) + "Lo" + str(HoV), arr=results)

