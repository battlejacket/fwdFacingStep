from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from statistics import mean 
import csv
from os import listdir

shortNameDict = {
    "data1800PlusPhysicsLambda1": "D+P$\_W_d1$",
    # "data1800PlusPhysicsLambda1": "D+P_L1",
    "data1800PlusPhysicsLambda01": "D+P$\_W_d0.1$",
    # "data1800PlusPhysicsLambda01": "D+P_L0.1",
    "dataOnly1800": "D",
    "physicsOnly": "P",
    "pressureDataPlusPhysicsLambda1": "PD+P",
    "2pO": "ToP"
}


resultsFilePath="./resultsL2.csv"
outputsPath="./outputs/fwdFacingStep/"
validatorSkip = ["DP5","DP36","DP79","DP86"] # skip data points
dirSkip = [".hydra", "init", "vtp"]

# models = ["data3600PlusPhysicsLambda05@500k", "data3600PlusPhysicsLambda1@500k", "physicsOnly@500k"]
models = listdir(outputsPath)
models.sort()

# models = ["physicsOnly@500k", "data1800PlusPhysicsLambda01@500k", "data1800PlusPhysicsLambda1@500k"]

with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")
    
    firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    writer.writerow(firstRow)
    
    for model in models:
        if model in dirSkip:
            # print("skipping ", model)
            continue
        # print("reading ", model)
        
        log_dir = outputsPath + model

        l2u = []
        l2v = []
        l2p = []
        names = []


        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()

        tags = event_accumulator.Tags()

        for tag in tags['tensors']:
            if 'Validators' in tag and not any(element in tag for element in validatorSkip):
                value = event_accumulator.Tensors(tag)[-1][-1].float_val[0]
                if 'error_u' in tag:
                    names.append(str(tag).split('/')[1].split('_')[0])
                    l2u.append(value)
                if 'error_v' in tag:
                    l2v.append(value)
                if 'error_p' in tag:
                    l2p.append(value)
                    
        l2uMax = (names[l2u.index(max(l2u))], max(l2u))
        l2vMax = (names[l2v.index(max(l2v))], max(l2v))
        l2pMax = (names[l2p.index(max(l2p))], max(l2p))

        l2uMin = (names[l2u.index(min(l2u))], min(l2u))
        l2vMin = (names[l2v.index(min(l2v))], min(l2v))
        l2pMin = (names[l2p.index(min(l2p))], min(l2p))

        l2uMean = mean(l2u)
        l2vMean = mean(l2v)
        l2pMean = mean(l2p)

        modelStrSplit = model.split("@")
                
        if len(modelStrSplit) == 3:
            label = shortNameDict[modelStrSplit[0]] + "@" + modelStrSplit[1].split("k")[0] + "k" + shortNameDict[modelStrSplit[1].split("k")[-1]] + "@" + modelStrSplit[-1]
        elif len(modelStrSplit) == 2:
            label = shortNameDict[modelStrSplit[0]] + "@" + modelStrSplit[-1]

        row = [label, l2uMean, l2uMin[1], str(l2uMax[1]) + " " + l2uMax[0], l2vMean, l2vMin[1], str(l2vMax[1]) + " " +  l2vMax[0], l2pMean, l2pMin[1], str(l2pMax[1]) + " " +  l2pMax[0]]
        writer.writerow(row)
        
        latexStr = label
            
        for value in row[1:]:
            # print(value)
            # valueF = round(float(value), 4)
            # latexStr += " & " + "%.4f" % valueF
            latexStr += " & " + str(value)
        latexStr += " \\\\"
        print(latexStr)