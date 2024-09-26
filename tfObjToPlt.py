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

models = ["physicsOnly@500k", "dataOnly1800@500k", "data1800PlusPhysicsLambda01@500k", "data1800PlusPhysicsLambda1@500k", "pressureDataPlusPhysicsLambda01@500k", "pressureDataPlusPhysicsLambda1@500k","data1800PlusPhysicsLambda01@100k2pO@500k", "data1800PlusPhysicsLambda1@100k2pO@500k"]

# models = ["physicsOnlyFC@500k", "dataOnly1800FC@500k", "data1800PlusPhysicsLambda01FC@500k", "data1800PlusPhysicsLambda1FC@500k", "pressureDataPlusPhysicsLambda01FC@500k", "pressureDataPlusPhysicsLambda1FC@500k"]


with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")
    
    firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    writer.writerow(firstRow)
    
    plt.figure(1)
    plt.title("Validation Error $p_2$")
    
    plt.figure(2)
    plt.title("Validation Error $p_1$")
    
    plt.figure(3)
    # plt.title("Validation Error $\Delta C_p$")
    plt.title("Fourier NN")
    # plt.title("Fully Connected NN")
    
    for model in models:
        # if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1] or '300k' in model.split("@")[-2]:
        if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
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
                    step = event.step
                    if step not in steps:
                        value = event.tensor_proto.float_val[0]
                        values.append(value)
                        steps.append(step)
                
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
            # print(key + ": ",trueUSPd[key])
            npUSP = np.array(USP[key])
            # print('npUSP' + ": ",npUSP)
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
        
        modelStrSplit = model.split("@")
                
        if len(modelStrSplit) == 3:
            label = shortNameDict[modelStrSplit[0]] + ", $S_d=$" + modelStrSplit[1].split("k")[0] + "k" #+ shortNameDict[modelStrSplit[1].split("k")[-1]] #+ "@" + modelStrSplit[-1]
        elif len(modelStrSplit) == 2:
            label = shortNameDict[modelStrSplit[0]] #+ "@" + modelStrSplit[-1]
        
        label = label.replace('Fully Connected, ', '').replace('Fourier, ', '')
        
        steps = np.array(steps)/1000
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
        plt.xlabel("Step ($x10^3$)")
        # plt.xlabel("Step")
        plt.ylabel("MAE $\Delta C_p$")
        plt.ylim(0.015, 10)
        
    
    # pre = 'TEST_'
    pre = 'F_'
    # pre = 'FC_'
    
    # plt.figure(1)    
    # plt.savefig(pre + "MaeDSP" + ".png", dpi = 600, bbox_inches='tight')
    # plt.figure(2)    
    # plt.savefig(pre + "MaeUSP" + ".png", dpi = 600, bbox_inches='tight')
    # plt.figure(3)    
    # plt.savefig(pre + "MaeDCp" + ".png", dpi = 600, bbox_inches='tight')
    
    plt.figure(1)    
    plt.savefig(pre + "MaeDSP" + ".svg", format='svg', dpi = 600, bbox_inches='tight')
    plt.figure(2)    
    plt.savefig(pre + "MaeUSP" + ".svg", format='svg', dpi = 600, bbox_inches='tight')
    plt.figure(3)    
    plt.savefig(pre + "MaeDCp" + ".svg", format='svg', dpi = 600, bbox_inches='tight')