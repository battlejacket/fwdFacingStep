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

# models = ["data1800PlusPhysicsLambda1@500k", "data1800PlusPhysicsLambda1FC@300k"]

with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")
    
    firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    writer.writerow(firstRow)
    
    plt.figure(1)
    plt.title("ME L2 u")
    
    plt.figure(2)
    plt.title("ME L2 v")
    
    plt.figure(3)
    plt.title("ME L2 p")
    
    for model in models:
        if model in dirSkip or "100k" in model.split("@")[-1] or "500k" in model.split("@")[-1]:
        # if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
        # if model in dirSkip:
            # print("skipping ", model)
            continue
        # print("reading ", model)
        
        log_dir = outputsPath + model

        # meanL2u = np.zeros(501)
        # meanL2v = np.zeros(501)
        # meanL2p = np.zeros(501)

        ea = EventAccumulator(log_dir, size_guidance={event_accumulator.TENSORS: 0})
        ea.Reload()

        tags = ea.Tags()

        n = 0
        
        
        for tag in tags['tensors']:
            if 'Validators' in tag and not any(element in tag for element in validatorSkip):
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

                if n == 0:
                    meanL2u = np.zeros(len(values))
                    meanL2v = np.zeros(len(values))
                    meanL2p = np.zeros(len(values))
                
                if 'error_u' in tag:
                    meanL2u += np.array(values)
                    n +=1
                    
                if 'error_v' in tag:
                    meanL2v += np.array(values)
                    
                if 'error_p' in tag:
                    meanL2p += np.array(values)
                    
        meanL2u /= n
        meanL2v /= n
        meanL2p /= n
    
        modelStrSplit = model.split("@")
                
        if len(modelStrSplit) == 3:
            label = shortNameDict[modelStrSplit[0]] + "@" + modelStrSplit[1].split("k")[0] + "k" + shortNameDict[modelStrSplit[1].split("k")[-1]] #+ "@" + modelStrSplit[-1]
        elif len(modelStrSplit) == 2:
            label = shortNameDict[modelStrSplit[0]] #+ "@" + modelStrSplit[-1]
        
        plt.figure(1)
        plt.plot(steps, meanL2u, label=label)
        plt.figure(2)
        plt.plot(steps, meanL2v, label=label)
        plt.figure(3)
        plt.plot(steps, meanL2p, label=label)
    
    for i in range(1,4):
        plt.figure(i)
        plt.legend()
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("ME L2")
    
    plt.figure(1)    
    plt.savefig("L2u" + ".png", dpi = 600)
    plt.figure(2)    
    plt.savefig("L2v" + ".png", dpi = 600)
    plt.figure(3)    
    plt.savefig("L2p" + ".png", dpi = 600)