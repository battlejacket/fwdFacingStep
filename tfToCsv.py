from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from statistics import mean 
import csv


resultsFilePath="./resultsL2.csv"
outputsPath="./outputs/fwdFacingStep"

with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")

    models = ["old_dataPlusPhysics3600@300k", "data3600PlusPhysicsLambda05@300k", "data3600PlusPhysicsLambda1@300k", "old_physicsOnly@500k", "physicsOnly@500k"]

    validatorSkip = ["DP5","DP36","DP79","DP86"]
    
    firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    
    writer.writerow(firstRow)
    

    for model in models:
        log_dir = "outputs/fwdFacingStep/" + model

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
                # names.append(str(tag).split('/')[1].split('_')[0])
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

        row = [model, l2uMean, l2uMin, l2uMax, l2vMean, l2vMin, l2vMax, l2pMean, l2pMin, l2pMax]
        writer.writerow(row)
        