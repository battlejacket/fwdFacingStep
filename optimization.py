import numpy as np
import dill
import os, glob, io, time
from os import listdir
import csv
from fwdFacingStep_fl import ffs, param_ranges, Re, Ho, Lo
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.visualization.scatter import Scatter
import contextlib
from multiprocessing import Process
from pymoo.termination.default import DefaultMultiObjectiveTermination

class modulusOptProblem(Problem):

    def __init__(self, n_var, n_obj, xl, xu, reNr, path):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.gen = 0
        self.reynoldsNr= reNr
        self.maxDesignsPerEvaluation = 100
        self.path = path
        # self.path = "./outputs/fwdFacingStep/data1800PlusPhysicsLambda01@500k"
        self.configFileDir = self.path+"/conf/"
        self.path_monitors = os.path.join(self.path, "monitors")

    def readFile(self, fileDir, objective, design):
        file = objective + "_design_" + str(design[0]) + ".csv"
        with open(os.path.join(fileDir, file), "r") as datafile:
            data = []
            reader = csv.reader(datafile, delimiter=",")
            for row in reader:
                columns = [row[1]]
                data.append(columns)
            last_row = float(data[-1][0])
            return np.array(last_row)

    def _evaluate(self, allDesigns, out, *args, **kwargs):
        start_time = time.time()
        if self.maxDesignsPerEvaluation > allDesigns.shape[0]:
            batches = 1
        else:
            batches = int(allDesigns.shape[0]/self.maxDesignsPerEvaluation)
        
        tfFiles = glob.glob(os.path.join(self.path, "events.out.tfevents*"))

        valuesF = []
        print("Generation " + str(self.gen) + ": Evaluating " + str(allDesigns.shape[0]) + " Designs in " + str(batches) + " Batches")
        for designs in np.array_split(ary=allDesigns, indices_or_sections=batches):
            # run modulus
            with contextlib.redirect_stdout(io.StringIO()):
                p = Process(target=ffs, args=(designs,self.reynoldsNr, self.configFileDir[2:], "config", True))
                p.start()
                p.join() 
            # read result files
            for design in enumerate(designs):
                # read upstream pressure
                objective = "upstreamPressure"
                USP = self.readFile(fileDir = self.path_monitors, objective = objective, design = design)
                # read downstream pressure
                objective = "downstreamPressure"
                DSP = self.readFile(fileDir = self.path_monitors, objective = objective, design = design)
                valuesF.append(2*(USP-DSP))
                
                # # read upstream pressureTot
                # objective = "upstreamPressureTot"
                # USPtot = self.readFile(fileDir = self.path_monitors, objective = objective, design = design)
                # # read downstream pressureTot
                # objective = "downstreamPressureTot"
                # DSPtot = self.readFile(fileDir = self.path_monitors, objective = objective, design = design)
                # valuesF.append(2*(USPtot-DSPtot))
                

            # remove old files
            filePattern = "*.csv"
            filePaths = glob.glob(os.path.join(self.path_monitors, filePattern))
            for file_path in filePaths:
                if "_design_" in file_path:
                    os.remove(file_path)
            
            filePattern = "events.out.tfevents*"
            filePaths = glob.glob(os.path.join(self.path, filePattern))
            for file_path in filePaths:
                if file_path not in tfFiles:
                    os.remove(file_path)

        out["F"] = np.array(valuesF)
        # out["F"] = out["F"].reshape(out["F"].shape[0], 1)
        # print(out["F"].shape)
        # print(out["F"])
        self.gen += 1
        elapsed_time = time.time() - start_time
        print("Evaluation time: ", elapsed_time)

# xl=np.array([0.25,float(param_ranges[Ho][0])])
# xu=np.array([0.95,float(param_ranges[Ho][1])])

xl=np.array([float(param_ranges[Lo][0]),float(param_ranges[Ho][0])])
xu=np.array([float(param_ranges[Lo][1]),float(param_ranges[Ho][1])])

outputsPath="./outputs/fwdFacingStep_fl/"
dirSkip = [".hydra", "init", "initFC", "data1800PlusPhysicsLambda1@500k", "dataOnly1800@500k"] #, "data1800PlusPhysicsLambda1FC@500k", "data1800PlusPhysicsLambda01FC@500k", "dataOnly1800FC@500k", "physicsOnlyFC@500k"]

# optResultsPath = "./optimizationResults/"
optResultsPath = "./optimizationResults_fl/"

doneModels = listdir(optResultsPath)

dirSkip += doneModels

# models = ["data1800PlusPhysicsLambda01@500k"]
models = listdir(outputsPath)
models.sort()
# models = ["physicsOnlyFC@500k", "physicsOnly@500k", "data1800PlusPhysicsLambda01@500k", ]

print("model list")
for model in models:
    if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
        continue 
    print(model)
print("end model list")

for model in models:
    if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
    # if model in dirSkip:
        print("skipping ", model)
        continue
        
    path = outputsPath + model
    optPath = optResultsPath + model
    
    reRange = range (300, 1100, 100)
    
    for reNr in reRange:

        if os.path.exists(optPath + "/optResultsX" + str(reNr) + ".npy"):
            print("skipping ", optPath + " " + str(reNr))
            continue
        
        optStartTime = time.time()
        
        print("Optimizing: ", str(model) + " " + str(reNr))
        
        problem = modulusOptProblem(n_var=2,n_obj=1, xl=xl, xu=xu, reNr=reNr, path=path)

        algorithm = DE(pop_size=200)

        termination = DefaultMultiObjectiveTermination(
            n_max_gen=1000, # default 1000
            n_max_evals=10000000
        )

        results = minimize(problem=problem, algorithm=algorithm,termination=termination)

        # with open("checkpoint", "wb") as f:
        #     dill.dump(algorithm, f)


        print("Optimization Done in ", time.time() - optStartTime)
        print("Best Design Objective Value: ", results.F)
        print("Best Design Parameter Value: ", results.X)

        if not os.path.exists(optPath):
            os.mkdir(optPath)
            
        np.save(file=optPath + "/popX" + str(problem.reynoldsNr), arr=results.pop.get("X"))
        np.save(file=optPath + "/popF" + str(problem.reynoldsNr), arr=results.pop.get("F"))

        np.save(file=optPath + "/optResultsF" + str(problem.reynoldsNr), arr=results.F)
        np.save(file=optPath + "/optResultsX" + str(problem.reynoldsNr), arr=results.X)