# shortNameDict = {
#     "data1800PlusPhysicsLambda1": "F D+P $W_d=1$",
#     # "data1800PlusPhysicsLambda1": "D+P_L1",
#     "data1800PlusPhysicsLambda01": "F D+P $W_d=0.1$",
#     # "data1800PlusPhysicsLambda01": "D+P_L0.1",
#     "dataOnly1800": "FNN D",
#     "dataOnly1800FC": "FC D",
#     "physicsOnly": "F P",
#     "physicsOnlyFC": "FC P",
#     "pressureDataPlusPhysicsLambda1": "F PD+P",
#     "pressureDataPlusPhysicsLambda1FC": "FC PD+P",
#     "data1800PlusPhysicsLambda1FC" : "FC D+P $W_d=1$",
#     "data1800PlusPhysicsLambda01FC" : "FC D+P $W_d=0.1$",
#     "physicsOnlynwlnshr": "F P NL", 
#     "physicsOnlyFCnwlnshr": "FC P NL", 
#     "2pO": " > P "
# }

shortNameDict = {
    "data1800PlusPhysicsLambda1": "FNN, Data+Physics, $W_d=1$",
    "data1800PlusPhysicsLambda01": "FNN, Data+Physics, $W_d=0.1$",
    "dataOnly1800": "FNN, Data",
    "dataOnly1800FC": "FCNN, Data",
    "physicsOnly": "FNN, Physics",
    "physicsOnlyFC": "FCNN, Physics",
    "pressureDataPlusPhysicsLambda1": "FNN, Pressure Data+Physics, $W_d=1$",
    "pressureDataPlusPhysicsLambda01": "FNN, Pressure Data+Physics, $W_d=0.1$",
    "pressureDataPlusPhysicsLambda1FC": "FCNN, Pressure Data+Physics, $W_d=1$",
    "pressureDataPlusPhysicsLambda01FC": "FCNN, Pressure Data+Physics, $W_d=0.1$",
    "data1800PlusPhysicsLambda1FC" : "FCNN, Data+Physics, $W_d=1$",
    "data1800PlusPhysicsLambda01FC" : "FCNN, Data+Physics, $W_d=0.1$",
    "2pO": " > Physics "
}

def name2data(name):
    data = {}
    if 'FC' in name:
        data['arch'] = 'FCNN'
    else:
        data['arch'] = 'FNN'
        
    if 'physicsOnly' in name:
        data['train'] = 'Physics'
    elif 'dataOnly' in name:
        data['train'] = 'Data'
    elif 'pressureData' in name and 'PlusPhysics' in name:
        data['train'] = 'Pressure Data+Physics'
    elif 'data' in name and 'PlusPhysics' in name:
        data['train'] = 'Data+Physics'
        
    if 'Lambda1' in name:
        data['Wd'] = '1'
    elif 'Lambda01' in name:
        data['Wd'] = '0.1'
    else:
        data['Wd'] = '-'
    
    # if len(name.split("@")) < 2:
    if '2pO' in name:
        # data['2P'] = True
        data['2PStep'] = name.split('@')[1].split('k')[0]
    elif 'data' in name:
        data['2PStep'] = '500'
    else:
        # data['2P'] = False
        data['2PStep'] = '-'
    return data