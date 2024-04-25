import os
from multiprocessing import Process
from fwdFacingStep import ffs
import shutil


valueList = [
    # {"useData": False, "usePhysics": True, "batchPerEpoch": 4000, "network_dir": "physicsOnlyBatches4000", "max_steps": 100000},
    # {"useData": False, "usePhysics": True, "batchPerEpoch": 4000, "network_dir": "physicsOnlyBatches4000", "max_steps": 300000},
    # {"useData": False, "usePhysics": True, "batchPerEpoch": 4000, "network_dir": "physicsOnlyBatches4000", "max_steps": 500000},
    # {"useData": False, "usePhysics": True, "batchPerEpoch": 4000, "decay_steps": 6000, "network_dir": "physicsOnlyBatches4000LowerDecay", "max_steps": 100000},
    # {"useData": False, "usePhysics": True, "batchPerEpoch": 4000, "decay_steps": 6000, "network_dir": "physicsOnlyBatches4000LowerDecay", "max_steps": 300000},
    # {"useData": False, "usePhysics": True, "batchPerEpoch": 4000, "decay_steps": 6000, "network_dir": "physicsOnlyBatches4000LowerDecay", "max_steps": 500000},
    {"useData": False, "usePhysics": True, "batchPerEpoch": 1000, "lr": "3e-4", "decay_steps": 6000, "network_dir": "physicsOnlyLowerLr", "max_steps": 100000},
    {"useData": False, "usePhysics": True, "batchPerEpoch": 1000, "lr": "3e-4", "decay_steps": 6000, "network_dir": "physicsOnlyLowerLr", "max_steps": 300000},
    {"useData": False, "usePhysics": True, "batchPerEpoch": 1000, "lr": "3e-4", "decay_steps": 6000, "network_dir": "physicsOnlyLowerLr", "max_steps": 500000},
            ]
baseConfigDir = "./conf/"
baseConfigFilePath = baseConfigDir + "config.yaml"
baseNetworkDir= "./outputs/fwdFacingStep/"
prevNetworkDir = ""

#loop over values (configurations to run) in value list
for values in valueList:
    with open(baseConfigFilePath, "r") as configFile:
        configFileList = configFile.readlines()
        #loop over lines in config file
        for i, line in enumerate(configFileList):
            #loop over items in current configuration
            for key, value in values.items():
                if key=="network_dir" and key in line.split(":")[0] and "initialization" not in line.split(":")[0]:
                    currentNetworkDir= value + "@" + str(int(values["max_steps"]/1000)) + "k"
                    configFileList[i]= line.split(":")[0] + ": \"" + currentNetworkDir + "\"\n"
                    if currentNetworkDir.split("@")[0]==prevNetworkDir.split("@")[0] and not os.path.exists(baseNetworkDir + currentNetworkDir):
                        #copy training progress to new dir
                        source_dir = baseNetworkDir + prevNetworkDir
                        destination_dir = baseNetworkDir + currentNetworkDir
                        shutil.copytree(source_dir, destination_dir)            
                elif key in line.split(":")[0] and "initialization" not in line.split(":")[0]:
                    configFileList[i]=line.split(":")[0] + ": " + str(value) + "\n"

    configFileDir = baseNetworkDir+currentNetworkDir+"/conf/"
    configFilePath = configFileDir + "config.yaml"
    if not os.path.exists(configFileDir):
        os.mkdir(baseNetworkDir+currentNetworkDir)
        os.mkdir(configFileDir)
        shutil.copy(baseConfigDir + "__init__.py", configFileDir + "__init__.py")
    
    with open(configFilePath, "w") as configFile:
        configFile.writelines(configFileList)
    
    p = Process(target=ffs, args=([],500,configFileDir[2:], "config"))
    p.start()
    p.join() 

    prevNetworkDir=currentNetworkDir


