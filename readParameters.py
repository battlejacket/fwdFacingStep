def setDict(parameters, parameterDict):
    for i, key in enumerate(parameterDict.keys()):
        parameterDict[key] = float(parameters[i])
    return parameterDict

def shortParameterString(parameters):
    shortString= ""
    for parameter in parameters:
        parameterF = round(float(parameter), 2)
        shortString += "%.2f" % parameterF + "-"
    return shortString[0:-1].replace(".", ",")

def readParametersFromFileName(fileName, parameterDict, generateNameString=False):
    parameters = fileName.split("_")[1].split(".")[0].replace(",", ".").split("-")
    parameterDict = setDict(parameters, parameterDict)
    if generateNameString:
        shortName = fileName.split("_")[0] + "_" + shortParameterString(parameters)
        return parameterDict, shortName
    else:
        return parameterDict

def readParametersFromCSV(row, parameterDict, generateNameString=False):
    nrParameters=len(parameterDict.keys())
    parameters = row[1:nrParameters+1]
    parameterDict = setDict(parameters, parameterDict)
    if generateNameString:
        shortName = row[0].replace(" ", "") + "_" + shortParameterString(parameters)
        return parameterDict, shortName
    else:
        return parameterDict
    
