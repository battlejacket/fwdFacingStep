from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from statistics import mean 
import csv
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from shortNames import shortNameDict


resultsFilePath="./resultsL2.csv"
outputsPath="./outputs/fwdFacingStep/"
validatorSkip = ["DP5","DP36","DP79","DP86"] # skip data points
# validatorSkip = [] # skip data points
dirSkip = [".hydra", "init", "vtp", "initFC"]

models = listdir(outputsPath)
models.sort()

# models = ["dataOnly1800FC@500k", "dataOnly1800@500k", "physicsOnlyFC@500k", "physicsOnly@500k"]
# models = ["data1800PlusPhysicsLambda1FC@500k", "data1800PlusPhysicsLambda1@500k", "data1800PlusPhysicsLambda01FC@500k", "data1800PlusPhysicsLambda01@500k"]
# models += ["data1800PlusPhysicsLambda1@100k2pO@500k", "data1800PlusPhysicsLambda01@100k2pO@500k"]
# models += ["pressureDataPlusPhysicsLambda1FC@500k", "pressureDataPlusPhysicsLambda1@500k"]

with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")
    
    firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    writer.writerow(firstRow)
    
    plt.figure(1)
    plt.title("MAE Downstream Pressure")
    
    plt.figure(2)
    plt.title("MAE Upstream Pressure")
    
    plt.figure(3)
    plt.title("MAE $\Delta C_p$")
    
    for model in models:
        if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1] or ('FC' not in model) or '300k' in model.split("@")[-2]:
        # if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
        # if model in dirSkip:
            # print("skipping ", model)
            continue
        print("reading ", model)
        
        log_dir = outputsPath + model

        DSP = {}
        trueDSPd = {}
        USP = {}
        trueUSPd = {}
        DCp = {}
        trueDCp = {}


        ea = EventAccumulator(log_dir, size_guidance={event_accumulator.TENSORS: 0})
        ea.Reload()

        tags = ea.Tags()

        n = 0
        
        
        for tag in tags['tensors']:
            if 'Monitors' in tag and not any(element in tag for element in validatorSkip) and not "_design_" in tag:
                values = []
                steps = []
                events = ea.Tensors(tag)
                pStep = -1
                for event in events:
                    # print(event)
                    step = event.step
                    if not step == pStep:
                        # print(step)
                        value = event.tensor_proto.float_val[0]
                        values.append(value)
                        steps.append(step)
                    pStep = step

                # if n == 0:
                    # meanDSP = np.zeros(len(values))
                    # meanUSP = np.zeros(len(values))
                    # meanDCp = np.zeros(len(values))
                
                if 'downstreamPressure' in tag:
                    DP = tag.split("DP")[1].split("_")[0]
                    trueDSPd[DP] = tag.split("=")[-1]
                    DSP[DP] = values
                    l = len(values)
                    n +=1
                
                if 'upstreamPressure' in tag:
                    DP = tag.split("DP")[1].split("_")[0]
                    trueUSPd[DP] = tag.split("=")[-1]
                    USP[DP] = values
                    
        
        meanUSP = np.zeros(l)
        meanDSP = np.zeros(l)
        meanDCp = np.zeros(l)
        
        for key in USP.keys():
            # print(key + ": ",trueUSP[key])
            npUSP = np.array(USP[key])
            trueUSP = float(trueUSPd[key])
            npUSPError = np.abs(npUSP - trueUSP)
            
            
            npDSP = np.array(DSP[key])
            trueDSP = float(trueDSPd[key])
            npDSPError = np.abs(npDSP - trueDSP)
            
            
            DCp = 2*(npUSP-npDSP)
            trueDCp = 2*(trueUSP-trueDSP)
            # print(key + ": ", trueDCp)
            npDCpError = np.abs(DCp-trueDCp)
            
            meanUSP += npUSPError
            meanDSP += npDSPError
            meanDCp += npDCpError
            

        meanUSP /= n
        meanDSP /= n
        meanDCp /= n
        
        
        # print(len(meanUSP))
        # print(len(meanDSP))
        # print(len(meanDCp))
        
    
        modelStrSplit = model.split("@")
                
        if len(modelStrSplit) == 3:
            label = shortNameDict[modelStrSplit[0]] + ", $S_d=$" + modelStrSplit[1].split("k")[0] + "k" #+ shortNameDict[modelStrSplit[1].split("k")[-1]] #+ "@" + modelStrSplit[-1]
        elif len(modelStrSplit) == 2:
            label = shortNameDict[modelStrSplit[0]] #+ "@" + modelStrSplit[-1]
        
        label = label.replace('Fully Connected, ', '').replace('Fourier, ', '')
        
        
        plt.figure(1)
        plt.plot(steps, meanUSP, label=label)
        plt.figure(2)
        plt.plot(steps, meanDSP, label=label)
        plt.figure(3)
        plt.plot(steps, meanDCp, label=label)
    
    for i in range(1,4):
        plt.figure(i)
        plt.legend()
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("MAE")
    
    plt.figure(1)    
    plt.savefig("MaeDSP" + ".png", dpi = 600)
    plt.figure(2)    
    plt.savefig("MaeUSP" + ".png", dpi = 600)
    plt.figure(3)    
    plt.savefig("MaeDCp" + ".png", dpi = 600)