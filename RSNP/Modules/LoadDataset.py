# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:21:46 2024

@author: ephouser
"""

from PythonModules import *

def loadData(datasetName, computer, expJ_Macros, seedTrainTestMic, shuffle = True):
    
    if computer == 0:
        fileName = "/home/ephouser/RSP/Data/"+datasetName+".csv"        
    elif computer == 1:
        gDrivePath = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\"
        fileName = gDrivePath+datasetName+".csv"
        
    dataTemp = pd.read_csv(fileName)
    dataFull = pd.DataFrame(data = dataTemp.values, columns = dataTemp.columns)
    dataFull.rename(columns = {"Response": "y"}, inplace = True)
    
    # sc = MinMaxScaler()
    # dataFull.iloc[:,:-1] = sc.fit_transform(dataFull.iloc[:,:-1])
    # dataFull = dataFull[dataFull.columns[(dataFull.var() != 0).values]]
    
    dataVars = dataFull.columns[:-1]
    dataFull[dataVars] = dataFull[dataVars].astype("float")
    dataFull.head()
    
    dataY = dataFull.loc[:,'y']
    dataX_All = dataFull.loc[:,dataFull.columns != 'y']
    
    # Shuffle Order of Columns
    if shuffle:
        varNames = list(dataX_All.columns)
        random.seed(1)
        dataX_All = dataX_All.loc[:,random.sample(varNames, len(varNames))]
    
    dataTrainTestMac = {}
    for EXP_J in expJ_Macros:
        # Build Train and Test Datasets
        temp_dataX_Train, temp_dataX_Test, temp_dataY_Train, temp_dataY_Test = train_test_split(dataX_All, dataY, test_size = 0.2, random_state= seedTrainTestMic[EXP_J])
        dataTrainTestMac[EXP_J] = {"trainX":temp_dataX_Train, "testX":temp_dataX_Test, 
                                "trainY":temp_dataY_Train, "testY":temp_dataY_Test}
    
    trueSol = [x for x in dataX_All.columns if 'x' in x]
    allFeats = dataX_All.columns
        
        
    return dataFull, dataY, dataX_All, allFeats, trueSol, dataTrainTestMac