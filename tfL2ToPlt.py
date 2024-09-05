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

# models = listdir(outputsPath)
# models.sort()

models = ["physicsOnly@500k", "dataOnly1800@500k", "data1800PlusPhysicsLambda01@500k", "data1800PlusPhysicsLambda1@500k", "pressureDataPlusPhysicsLambda01@500k", "pressureDataPlusPhysicsLambda1@500k",
"data1800PlusPhysicsLambda01@100k2pO@500k", "data1800PlusPhysicsLambda1@100k2pO@500k"]


# models = ["physicsOnlyFC@500k", "dataOnly1800FC@500k", "data1800PlusPhysicsLambda01FC@500k", "data1800PlusPhysicsLambda1FC@500k", "pressureDataPlusPhysicsLambda01FC@500k", "pressureDataPlusPhysicsLambda1FC@500k"] #, "data1800PlusPhysicsLambda1FC@100k2pO@500k"]


with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")
    
    firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    writer.writerow(firstRow)
    
    plt.figure(1)
    plt.title("Validation Error $U$")
    
    plt.figure(2)
    plt.title("Validation Error $V$")
    
    plt.figure(3)
    plt.title("Validation Error $P$")
    
    for model in models:
        if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1] or '300k' in model.split("@")[-2]:
        # if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
        # if model in dirSkip:
            # print("skipping ", model)
            continue
        print("reading ", model)
        
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

                for event in events:
                    step = event.step
                    if step not in steps:
                        value = event.tensor_proto.float_val[0]
                        values.append(value)
                        steps.append(step)

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
            label = shortNameDict[modelStrSplit[0]] + ", $S_d=$" + modelStrSplit[1].split("k")[0] + "k" #+ shortNameDict[modelStrSplit[1].split("k")[-1]] #+ "@" + modelStrSplit[-1]
        elif len(modelStrSplit) == 2:
            label = shortNameDict[modelStrSplit[0]] #+ "@" + modelStrSplit[-1]
        
        label = label.replace('Fully Connected, ', '').replace('Fully Conn., ', '').replace('Fourier, ', '')
        
        steps = np.array(steps)/1000
        plt.figure(1)
        # ax = plt.subplot(111)
        plt.plot(steps, meanL2u, label=label)
        # ax.plot(steps, meanL2u, label=label)
        # ax.legend(bbox_to_anchor=(2, 2))
        plt.figure(2)
        plt.plot(steps, meanL2v, label=label)
        plt.figure(3)
        plt.plot(steps, meanL2p, label=label)
    
    for i in range(1,4):
        plt.figure(i)
        # plt.legend()
        plt.yscale("log")
        plt.xlabel("Step ($x10^3$)")
        plt.ylabel("Mean $L^2$ Error")
        plt.ylim(0.003, 2)
        
    pre= 'F_'
    
    plt.figure(1)    
    plt.savefig(pre + "L2u" + ".png", dpi = 600, bbox_inches='tight')
    plt.figure(2)    
    plt.savefig(pre + "L2v" + ".png", dpi = 600, bbox_inches='tight')
    plt.figure(3)    
    plt.savefig(pre + "L2p" + ".png", dpi = 600, bbox_inches='tight')