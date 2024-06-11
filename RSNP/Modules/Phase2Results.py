# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:51:33 2024

@author: ephouser
"""

from PythonModules import *                   # General Python Modules (numpy, pandas, etc..)
from EstimatingPerformance import *           # Module for Measuring Performance of Solutions

#####################
# GET PHASE 2 RESULTS
def getP2Results(computer, datasetName, expSol, nREP, seedTrainTest):
    
    #############
    # IMPORT DATA    
    if computer == 0:
        P2_fileName = "/home/ephouser/RSP/Data/"+datasetName+".csv"        
    else:
        gDrivePath = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\"
        P2_fileName = gDrivePath + datasetName + ".csv"   
            
    data_P2 = pd.read_csv(P2_fileName)
    dataY_P2 = data_P2.loc[:,"y"]                        # Response Data
    dataX_P2 = data_P2.loc[:,data_P2.columns != "y"]     # Explanatory Data
    allFeats_P2 = list(dataX_P2.columns)
    
    ######################
    # SHUFFLE COLUMN ORDER
    shuffle = True
    if shuffle:
        random.seed(7)
        dataX_P2 = dataX_P2.loc[:,random.sample(allFeats_P2, len(allFeats_P2))]
    
    ##################################
    # GENERATE TRAIN AND TEST DATASETS
    dataTrainTest_P2 = {}
    for r in range(nREP):
        # Build Train and Test Datasets
        trainX, testX, trainY, testY = train_test_split(dataX_P2, dataY_P2, test_size = 0.2, random_state= seedTrainTest[r])
        dataTrainTest_P2[r] = {"trainX":trainX, "testX":testX, "trainY":trainY, "testY":testY}
                    
    tempSol = np.array(expSol)
    tempYbar = []
    tempYij = []
    tempVar = []
    
    if len(expSol) > 0:
        # For Each Replication
        for EXP_P2 in range(nREP):
            
            trainX_P2 = dataTrainTest_P2[EXP_P2]['trainX']
            testX_P2 = dataTrainTest_P2[EXP_P2]['testX']
            trainY_P2 = dataTrainTest_P2[EXP_P2]['trainY']
            testY_P2  = dataTrainTest_P2[EXP_P2]['testY']
            
            tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX_P2.loc[:,expSol],
                                                      testX_P2.loc[:,expSol],
                                                      trainY_P2,
                                                      testY_P2)
            tempYij.append(tempRMSE)
        
        tempYij = np.array(tempYij)
        tempYbar = round((1/len(tempYij))*sum(tempYij),5)
        tempVar = round(sum((tempYij-tempYbar)**2)/(nREP-1) ,5)

    return tempYij, tempYbar, tempVar, allFeats_P2


