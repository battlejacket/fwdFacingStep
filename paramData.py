from readParameters import readParametersFromFileName
from datasetFromCsv import datasetFromCsv
from sympy import Symbol
from os import walk, path
from modulus.sym.hydra import to_absolute_path


Re = Symbol("Re")
x, y = Symbol("x"), Symbol("y")
Lo, Ho = Symbol("Lo"), Symbol("Ho")

# global ranges for my parameters
param_ranges = {
    Re: (100, 1000),
    Lo: (0.1, 1),
    Ho: (0.1, 0.5),
    }


ansysInvarNames = ["Points:0", "Points:1"]
modulusInvarNames = ["x", "y"]
ansysOutvarNames = ["Pressure", "Velocity:0", "Velocity:1"]
modulusOutvarNames = ["p_d", "u_d", "v_d"] # these are custom node i added to separate the data loss from other losses with u,v,p they are simply defined as 1*u etc, look line 189 in fwdFacingStep
scales = {"p_d": (0,1), "u_d": (0,1), "v_d": (0,1), "x": (0,1), "y": (-0.5,1)} #used to scale and translate ansys data to mach the modulus setup. in this case all variables use a scale of one and only y is translated to be centered around y=0 instead of y=0.5
lambdaWeighting = {"p_d": 0.1, "u_d": 0.1, "v_d": 0.1}
additionalConstraints=None #{"continuity": 0, "momentum_x": 0, "momentum_y": 0} # can be used to add other variables not in the csv to the output vector
criteria=None # if you want to use a sympy criteria to for example only use data for half the domain, same as other modulus criterias

# loop through all csv files in data directory.
for root, dirs, files in walk(to_absolute_path("./ansys/dataT")):
    for i, name in enumerate(files):
        # read parmater values for the specific csv from its file name, requires it to be named using name_valueParameter1-valueParameter2-valueParameter3.csv
        # any number of parameters can be used but must correspond to the ones in the global parameter range dictionary (param_ranges) in number and order.
        dataParameterRange, shortName = readParametersFromFileName(fileName=name, parameterDict=param_ranges, generateNameString=True)
        #dataParameterRange contains the specific parameter values, shortName is just the file name with fewer digits for each parameter.
        
        print("parameter values for file " + shortName + " :", dataParameterRange)
        
        filePath = str(path.join(root, name))
        
        # generate a data set to be used with pointwiseConstraint.from_numpy()
        # first a non parameterized version for reference 
        dataInvar_noParam, dataOutvar_noParam, lambdaWeights_noParam = datasetFromCsv(
            filePath=filePath,
            csvInvarNames=ansysInvarNames,
            csvOutvarNames=ansysOutvarNames,
            modulusInvarNames=modulusInvarNames,
            modulusOutvarNames=modulusOutvarNames,
            scales=scales,
            parameterRanges=None,
            criteria=criteria,
            additionalConstraints=additionalConstraints,
            lambdaWeighting=lambdaWeighting,
            )
        
        print("\n\n data set for " + shortName + ", non parameterized version:")
        print("\t variables are stored in a dictionary with a key for each variable, input variables are: ", dataInvar_noParam.keys())
        print("\t first 5 lines for each input variable:")
        for key in dataInvar_noParam.keys():
            print("\t key " + str(key) + ":")
            for line in dataInvar_noParam[key][0:5]:
                print("\t\t" + str(line).strip("[]"))

        
        print("\t output variables in the data set are: ", dataOutvar_noParam.keys())
        print("\t first 5 lines for each output variable:")
        for key in dataOutvar_noParam.keys():
            print("\t key " + str(key) + ":")
            for line in dataOutvar_noParam[key][0:5]:
                print("\t\t" + str(line).strip("[]"))
        print("t during training the data will be structured like this input vector = output vector and used as NN(input vetctor) should equal output vector and the deviation from that will be used to formulate the loss. During training it will also be mixed with the rows from other constraints and order of the rows will be shuffled/randomized:")
        for i, line in enumerate(zip(zip(dataInvar_noParam["x"], dataInvar_noParam["y"]), zip(dataOutvar_noParam["u_d"], dataOutvar_noParam["v_d"], dataOutvar_noParam["p_d"]))):
            if i > 5:
                continue
            else:
                lineIn=str(line[0][0][0]) + ",\t" + str(line[0][1][0])
                lineOut=str(line[1][0][0]) + ",\t" + str(line[1][1][0])
                print(str(lineIn) + "=" + str(lineOut))

        dataInvar, dataOutvar, lambdaWeights = datasetFromCsv(
            filePath=filePath,
            csvInvarNames=ansysInvarNames,
            csvOutvarNames=ansysOutvarNames,
            modulusInvarNames=modulusInvarNames,
            modulusOutvarNames=modulusOutvarNames,
            scales=scales,
            parameterRanges=dataParameterRange,
            criteria=criteria,
            additionalConstraints=additionalConstraints,
            lambdaWeighting=lambdaWeighting,
            )
        
        print("\n\n data set for " + shortName + ", parameterized version:")
        print("\t variables are stored in a dictionary with a key for each variable, input variables are: ", dataInvar.keys())
        print("\t first 5 lines for each input variable:")
        for key in dataInvar.keys():
            print("\t key " + str(key) + ":")
            for line in dataInvar[key][0:5]:
                print("\t\t" + str(line).strip("[]"))

        
        print("\t output variables in the data set are: ", dataOutvar.keys())
        print("\t first 5 lines for each output variable:")
        for key in dataOutvar.keys():
            print("\t key " + str(key) + ":")
            for line in dataOutvar[key][0:5]:
                print("\t\t" + str(line).strip("[]"))
        print("t during training the data will be structured like this input vector = output vector and used as NN(input vetctor) should equal output vector and the deviation from that will be used to formulate the loss. During training it will also be mixed with the rows from other constraints and order of the rows will be shuffled/randomized:")
        for i, line in enumerate(zip(zip(dataInvar["x"], dataInvar["y"], dataInvar["Re"], dataInvar["Lo"], dataInvar["Ho"]), zip(dataOutvar["u_d"], dataOutvar["v_d"], dataOutvar["p_d"]))):
            if i > 5:
                continue
            else:
                lineIn=str(line[0][0][0]) + ",\t" + str(line[0][1][0]) + ",\t" + str(line[0][2][0]) + ",\t" + str(line[0][3][0]) + ",\t" + str(line[0][4][0])
                lineOut=str(line[1][0][0]) + ",\t" + str(line[1][1][0])
                print(str(lineIn) + "=" + str(lineOut))
        print("each row represents one point/node in the domain/exported csv, the (x,y) and (u,v,p) values varies depending on location in the domain while the parameter values are the same for each row since all rows/points in the csv are from the same case, this is also why the data contraints are only applied for the specific parameter values, modulus will not change the parameter values for any of these rows")

# the constraint would then be added like this, look at line 377 in fwdFacingStep
                            
# dataConstraint = PointwiseConstraint.from_numpy(
#     nodes=nodes, 
#     invar=dataInvar, 
#     outvar=dataOutvar, 
#     batch_size=int(dataInvar['x'].size/batches),
#     lambda_weighting=lambdaWeights
# )

# domain.add_constraint(dataConstraint, shortName)