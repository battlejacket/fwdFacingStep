from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from statistics import mean, stdev
import csv
from os import listdir
from shortNames import name2data #shortNameDict

resultsFilePath="./resultsL2.csv"
outputsPath="./outputs/fwdFacingStep/"
validatorSkip = ["DP5","DP36","DP79","DP86"] # skip data points
dirSkip = [".hydra", "init", "initFC", "vtp"]

models = listdir(outputsPath)
models.sort()

# models = ["physicsOnlynwlnshr@500k", "physicsOnlyFCnwlnshr@500k", "physicsOnly@500k", "physicsOnlyFC@500k"]

with open(resultsFilePath, "w") as resultsFile:
    writer = csv.writer(resultsFile, delimiter=",")
    
    firstRow = ["model", "u mean", "u std", "v mean", "v std", "p mean", "p std"]
    # firstRow = ["model", "u mean", "u min", "u max", "v mean", "v min", "v max", "p mean", "p min", "p max"]
    writer.writerow(firstRow)
    data = []
    
    for model in models:
        if model in dirSkip or "100k" in model.split("@")[-1] or "300k" in model.split("@")[-1]:
        # if model in dirSkip:
            # print("skipping ", model)
            continue
        print("reading ", model)
        
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

        l2uStd = stdev(l2u)
        l2vStd = stdev(l2v)
        l2pStd = stdev(l2p)

        # modelStrSplit = model.split("@")
                
        # if len(modelStrSplit) == 3:
        #     label = shortNameDict[modelStrSplit[0]] + shortNameDict[modelStrSplit[1].split("k")[-1]] + "@" + modelStrSplit[1].split("k")[0] + "k"#+ "@" + modelStrSplit[-1]
        # elif len(modelStrSplit) == 2:
        #     label = shortNameDict[modelStrSplit[0]] # + "@" + modelStrSplit[-1]

        # # row = [label, l2uMean, l2uMin[1], str(l2uMax[1]) + " " + l2uMax[0], l2vMean, l2vMin[1], str(l2vMax[1]) + " " +  l2vMax[0], l2pMean, l2pMin[1], str(l2pMax[1]) + " " +  l2pMax[0]]
        # row = [label, l2uMean, l2uStd, l2vMean, l2vStd, l2pMean, l2pStd]
        # writer.writerow(row)
        
        # latexStr = label
            
        # for value in row[1:]:
        #     # print(value)
        #     valueF = round(float(value), 3)
        #     latexStr += " & " + "%.3f" % valueF
        #     # latexStr += " & " + str(value)
        # latexStr += " \\\\"
        # print(latexStr)
        
        modelData = name2data(model)
        
        # modelData[]
            
        # row = [label, meanDSP, minDSP, maxDSP, meanUSP, minUSP, maxUSP, meanDCp, minDCp, maxDCp]
        row = [modelData['train'], modelData['Wd'], modelData['2PStep'], modelData['arch'], l2uMean, l2uStd, l2vMean, l2vStd, l2pMean, l2pStd]
        
        
        data.append(row)
        
    # popsorted=pop[pop[:, -1].argsort()]

    # dataSorted = data[data[:,0].argsort()]
    dataSorted = data


    print(data)
    
    for row in dataSorted:
        
        writer.writerow(row)
        # latexStr = label
        latexStr = ''
        for value in row[0:3]:
            # print('f ',value)
            latexStr += str(value) + ' & '
        latexStr += str(row[3])
        for value in row[4:]:
            # print('d ', value)
            valueF = round(float(value), 3)
            # latexStr += "%.3f" % valueF
            latexStr += " & " + "%.3f" % valueF
        latexStr += " \\\\"
        print(latexStr) 