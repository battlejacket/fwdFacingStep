import os
from multiprocessing import Process
from fwdFacingStep import ffs
import shutil

valueList = [
    # physicsOnly (15.5h)
    # {"useData": False, "usePhysics": True, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "physicsOnlyFC", "max_steps": 100000},
    # {"useData": False, "usePhysics": True, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "physicsOnlyFC", "max_steps": 300000},
    # {"useData": False, "usePhysics": True, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "physicsOnlyFC", "max_steps": 500000},
    
    # data1800PlusPhysicsLambda1FC (est 32h)
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "data1800PlusPhysicsLambda1FC",  "max_steps": 100000},
    {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "data1800PlusPhysicsLambda1FC", "max_steps": 300000},
    {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "data1800PlusPhysicsLambda1FC", "max_steps": 500000},
    # data1800PlusPhysicsLambda01FC (est 32h)
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "data1800PlusPhysicsLambda01FC", "max_steps": 100000},
    {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "data1800PlusPhysicsLambda01FC", "max_steps": 300000},
    {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "netArch": "FC", "initialization_network_dir": "initFC", "network_dir": "data1800PlusPhysicsLambda01FC", "max_steps": 500000},
    
    # # physicsOnly (15.5h)
    # {"useData": False, "usePhysics": True, "network_dir": "physicsOnly", "max_steps": 100000},
    # {"useData": False, "usePhysics": True, "network_dir": "physicsOnly", "max_steps": 300000},
    # {"useData": False, "usePhysics": True, "network_dir": "physicsOnly", "max_steps": 500000},
    # # data1800PlusPhysicsLambda01 (est 32h)
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "network_dir": "data1800PlusPhysicsLambda01", "max_steps": 100000},
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "network_dir": "data1800PlusPhysicsLambda01", "max_steps": 300000},
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "network_dir": "data1800PlusPhysicsLambda01", "max_steps": 500000},
    # # data1800PlusPhysicsLambda1 (est 32h)
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "network_dir": "data1800PlusPhysicsLambda1", "max_steps": 100000},
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "network_dir": "data1800PlusPhysicsLambda1", "max_steps": 300000},
    # {"useData": True, "usePhysics": True, "batchesData": 1800, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "network_dir": "data1800PlusPhysicsLambda1", "max_steps": 500000},
    
    # # dataOnly1800 (est <8h)
    # {"useData": True, "usePhysics": False, "batchesData": 1800, "network_dir": "dataOnly1800", "max_steps": 100000},
    # {"useData": True, "usePhysics": False, "batchesData": 1800, "network_dir": "dataOnly1800", "max_steps": 300000},
    # {"useData": True, "usePhysics": False, "batchesData": 1800, "network_dir": "dataOnly1800", "max_steps": 500000},
    
    # # pressureDataPlusPhysicsLambda01 (est 20h)
    # {"useData": True, "usePhysics": True, "pressureDataOnly": True, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "network_dir": "pressureDataPlusPhysicsLambda01", "max_steps": 100000},
    # {"useData": True, "usePhysics": True, "pressureDataOnly": True, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "network_dir": "pressureDataPlusPhysicsLambda01", "max_steps": 300000},
    # {"useData": True, "usePhysics": True, "pressureDataOnly": True, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "network_dir": "pressureDataPlusPhysicsLambda01", "max_steps": 500000},
    # # pressureDataPlusPhysicsLambda1 (est 20h)
    # {"useData": True, "usePhysics": True, "pressureDataOnly": True, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "network_dir": "pressureDataPlusPhysicsLambda1", "max_steps": 100000},
    # {"useData": True, "usePhysics": True, "pressureDataOnly": True, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "network_dir": "pressureDataPlusPhysicsLambda1", "max_steps": 300000},
    # {"useData": True, "usePhysics": True, "pressureDataOnly": True, "lambda_p_d": 1, "lambda_u_d": 1, "lambda_v_d": 1, "network_dir": "pressureDataPlusPhysicsLambda1", "max_steps": 500000},
    
    # data3600PlusPhysicsLambda01 (est 24h)
    # {"useData": True, "usePhysics": True, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "network_dir": "data3600PlusPhysicsLambda01", "max_steps": 100000},
    # {"useData": True, "usePhysics": True, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "network_dir": "data3600PlusPhysicsLambda01", "max_steps": 300000},
    # {"useData": True, "usePhysics": True, "lambda_p_d": 0.1, "lambda_u_d": 0.1, "lambda_v_d": 0.1, "network_dir": "data3600PlusPhysicsLambda01", "max_steps": 500000},
    # data3600PlusPhysicsLambda1 (est 24h)
    # {"useData": True, "usePhysics": True, "lambda_p_d": 1,   "lambda_u_d": 1,   "lambda_v_d": 1, "network_dir": "data3600PlusPhysicsLambda1", "max_steps": 100000},
    # {"useData": True, "usePhysics": True, "lambda_p_d": 1,   "lambda_u_d": 1,   "lambda_v_d": 1, "network_dir": "data3600PlusPhysicsLambda1", "max_steps": 300000},
    # {"useData": True, "usePhysics": True, "lambda_p_d": 1,   "lambda_u_d": 1,   "lambda_v_d": 1, "network_dir": "data3600PlusPhysicsLambda1", "max_steps": 500000},
    
    # dataOnly900
    # {"useData": True, "usePhysics": False, "batchesData": 900, "network_dir": "dataOnly900", "max_steps": 100000},
    # {"useData": True, "usePhysics": False, "batchesData": 900, "network_dir": "dataOnly900", "max_steps": 300000},
    # {"useData": True, "usePhysics": False, "batchesData": 900, "network_dir": "dataOnly900", "max_steps": 500000},
    
    # data1800PlusPhysicsLambda1@300k2pO
    # {"useData": False, "usePhysics": True, "network_dir": "data1800PlusPhysicsLambda1@300k2pO", "max_steps": 500000},
    
    # data1800PlusPhysicsLambda1@100k2pO
    # {"useData": False, "usePhysics": True, "network_dir": "data1800PlusPhysicsLambda1@100k2pO", "max_steps": 300000},
    # {"useData": False, "usePhysics": True, "network_dir": "data1800PlusPhysicsLambda1@100k2pO", "max_steps": 500000},
    
    # data1800PlusPhysicsLambda01@100k2pO
    # {"useData": False, "usePhysics": True, "network_dir": "data1800PlusPhysicsLambda01@100k2pO", "max_steps": 300000},
    # {"useData": False, "usePhysics": True, "network_dir": "data1800PlusPhysicsLambda01@100k2pO", "max_steps": 500000},
    
    # data1800PlusPhysicsLambda01@300k2pO
    # {"useData": False, "usePhysics": True, "network_dir": "data1800PlusPhysicsLambda01@300k2pO", "max_steps": 500000},
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
                elif key == "initialization_network_dir" and key in line.split(":")[0]:
                    configFileList[i]=line.split(":")[0] + ": " + str(value) + "\n"
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
    
    p = Process(target=ffs, args=([],500,configFileDir[2:], "config", False))
    p.start()
    p.join() 

    prevNetworkDir=currentNetworkDir


