shortNameDict = {
    "data1800PlusPhysicsLambda1": "F D+P $W_d=1$",
    # "data1800PlusPhysicsLambda1": "D+P_L1",
    "data1800PlusPhysicsLambda01": "F D+P $W_d=0.1$",
    # "data1800PlusPhysicsLambda01": "D+P_L0.1",
    "dataOnly1800": "F D",
    "dataOnly1800FC": "FC D",
    "physicsOnly": "F P",
    "physicsOnlyFC": "FC P",
    "pressureDataPlusPhysicsLambda1": "F PD+P",
    "pressureDataPlusPhysicsLambda1FC": "FC PD+P",
    "data1800PlusPhysicsLambda1FC" : "FC D+P $W_d=1$",
    "data1800PlusPhysicsLambda01FC" : "FC D+P $W_d=0.1$",
    "physicsOnlynwlnshr": "F P NL", 
    "physicsOnlyFCnwlnshr": "FC P NL", 
    "2pO": " > P "
}

def name2data(name):
    data = {}
    if 'FC' in name:
        data['arch'] = 'FC'
    else:
        data['arch'] = 'F'
        
    if 'physicsOnly' in name:
        data['train'] = 'P'
    elif 'dataOnly' in name:
        data['train'] = 'D'
    elif 'pressureData' in name and 'PlusPhysics' in name:
        data['train'] = 'PD+P'
    elif 'data' in name and 'PlusPhysics' in name:
        data['train'] = 'D+P'
        
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