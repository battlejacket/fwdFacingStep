import numpy as np
import dill
import os, glob, io, time
import csv
from thermal import modHeat
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
import contextlib
from multiprocessing import Process
from pymoo.termination.default import DefaultMultiObjectiveTermination

from globalProperties import *

# def save_dict(di_, filename_):
#     with open(filename_, 'wb') as f:
#         pickle.dump(di_, f)

# def load_dict(filename_):
#     with open(filename_, 'rb') as f:
#         ret_di = pickle.load(f)
#     return ret_di

# if __name__ == '__main__':
#     g_data = {
#         'm':np.random.rand(4,4),
#         'n':np.random.rand(2,2,2)
#     }
#     save_dict(g_data, './data.pkl')
#     g_data2 = load_dict('./data.pkl')
#     print(g_data['m'] == g_data2['m'])
#     print(g_data['n'] == g_data2['n'])

class modulusOptProblem(Problem):

    def __init__(self, n_var=-1, n_obj=1, n_ieq_constr=0, n_eq_constr=0, xl=None, xu=None, vtype=None, vars=None, elementwise=False, elementwise_func=..., elementwise_runner=..., replace_nan_values_by=None, exclude_from_serialization=None, callback=None, strict=True, **kwargs):
        super().__init__(n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu, vtype, vars, elementwise, elementwise_func, elementwise_runner, replace_nan_values_by, exclude_from_serialization, callback, strict, **kwargs)
        self.gen = 0
        self.maxDesignsPerEvaluation = 35
        self.path_thermal = "./outputs/thermal/rsF2/"
        self.path_monitors = os.path.join(self.path_thermal, "monitors")

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
        strat_time = time.time()
        if self.maxDesignsPerEvaluation > allDesigns.shape[0]:
            batches = 1
        else:
            batches = int(allDesigns.shape[0]/self.maxDesignsPerEvaluation)
        
        tfFiles = glob.glob(os.path.join(self.path_thermal, "events.out.tfevents*"))

        valuesF = []
        valuesG = []
        print("Generation " + str(self.gen) + ": Evaluating " + str(allDesigns.shape[0]) + " Designs in " + str(batches) + " Batches")
        for designs in np.array_split(ary=allDesigns, indices_or_sections=batches):
            # run modulus
            # with contextlib.redirect_stdout(io.StringIO()):
            p = Process(target=modHeat, args=(designs,))
            p.start()
            p.join() 
            # read result files
            for design in enumerate(designs):
                # # read solid max temp
                # objective = "solidMaxTemp"
                # valuesF.append(self.readFile(fileDir = self.path_monitors, objective = objective, design = design))
                
                # read solid ave temp
                objective = "solidAveTemp"
                valuesF.append(self.readFile(fileDir = self.path_monitors, objective = objective, design = design))

                # read solid max temp grad
                objective = "solidMaxTempGrad"
                valuesF.append(self.readFile(fileDir = self.path_monitors, objective = objective, design = design))
                
                # # read solid ave temp grad
                # objective = "solidAveTempGrad"
                # valuesF.append(self.readFile(fileDir = self.path_monitors, objective = objective, design = design))
                
                # read cooling inlet flow 1
                objective = "coolingInletFlow1"
                inletFlow1 = self.readFile(fileDir = self.path_monitors, objective = objective, design = design)
                objective = "coolingInletFlow2"
                inletFlow2 = self.readFile(fileDir = self.path_monitors, objective = objective, design = design)
                valuesG.append(inletFlow1 + inletFlow2-0.001)
                

            # # remove old files
            # filePattern = "*.csv"
            # filePaths = glob.glob(os.path.join(self.path_monitors, filePattern))
            # for file_path in filePaths:
            #     os.remove(file_path)
            
            # filePattern = "events.out.tfevents*"
            # filePaths = glob.glob(os.path.join(self.path_thermal, filePattern))
            # for file_path in filePaths:
            #     if file_path not in tfFiles:
            #         os.remove(file_path)


        out["F"] = np.array(valuesF)
        out["G"] = np.array(valuesG)
        self.gen += 1
        elapsed_time = time.time() - strat_time
        print("Evaluation time: ", elapsed_time)
        
        
xl=np.array([paramRang[holeRadius1][0][0],paramRang[holeRadius2][0][0],paramRang[inletVelocity1][0][0],paramRang[inletVelocity2][0][0],paramRang[holePosX1][0][0],paramRang[holePosX2][0][0],paramRang[holePosZ1][0][0],paramRang[holePosZ2][0][0]])
xu=np.array([paramRang[holeRadius1][0][0],paramRang[holeRadius2][0][1],paramRang[inletVelocity1][0][1],paramRang[inletVelocity2][0][1],paramRang[holePosX1][0][1],paramRang[holePosX2][0][1],paramRang[holePosZ1][0][1],paramRang[holePosZ2][0][1]])

problem = modulusOptProblem(n_var=8,n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

algorithm = NSGA2(pop_size=500)

termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=50, # default 1000
    n_max_evals=100000
)

results = minimize(problem=problem, algorithm=algorithm,termination=termination)

# with open("checkpoint", "wb") as f:
#     dill.dump(algorithm, f)


print("Optimization Done!")
# print("Best Design Objective Value: ", results.F)
# print("Best Design Objective Value Plot: ", zip(results.F[0], results.F[1]))
# print("Best Design Parameter Value: ", results.X)

np.save(file="tpopX", arr=results.pop.get("X"))
np.save(file="tpopF", arr=results.pop.get("F"))
np.save(file="tpopG", arr=results.pop.get("G"))

np.save(file="toptResultsF", arr=results.F)
np.save(file="toptResultsX", arr=results.X)
np.save(file="toptResultsG", arr=results.G)