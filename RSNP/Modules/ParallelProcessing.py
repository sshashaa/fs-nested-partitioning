# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:39:39 2024

@author: ephouser
"""

from PythonModules import *

def Xprocess_iteration(processData):

    j, numReps, numNewReps, dataTrainValidMic, tempNewSol = processData

    if j > (numReps - numNewReps - 1):
        temp_trainX = dataTrainValidMic[j]['trainX']
        temp_testX = dataTrainValidMic[j]['testX']
        temp_trainY = dataTrainValidMic[j]['trainY']
        temp_testY = dataTrainValidMic[j]['testY']

        tempMAE, tempMSE, tempRMSE = fitLinearReg(temp_trainX.loc[:, tempNewSol == 1],
                                                  temp_testX.loc[:, tempNewSol == 1],
                                                  temp_trainY,
                                                  temp_testY)
        return -1 * tempRMSE
    else:
        return 0 


def process_iteration(i, r, repInfo, Ni, X, tempYij, tempYbari, modelType, dataTrainValidMic):

    tempNewSol = np.array(X[i])
    numPrevReps, numReps, numNewReps  = repInfo  

    if sum(tempNewSol) == 0:
        tempYij = [-1*99999 for _ in range(numNewReps)]
        tempYbari.append(-1*99999)
    else:
        if modelType == 0:
            if numReps > len(dataTrainValidMic):
                dataTrainValidMic = getMoreTrainTestData(dataX_Train_MacJ, dataY_Train_MacJ, testSize,
                                                         numReps - len(dataTrainValidMic), seedTrainTest, dataTrainValidMic)
            temptempYij = []
            for j in range(numReps):
                if j > (numReps - numNewReps - 1):
                    temp_trainX = dataTrainValidMic[j]['trainX']
                    temp_testX = dataTrainValidMic[j]['testX']
                    temp_trainY = dataTrainValidMic[j]['trainY']
                    temp_testY = dataTrainValidMic[j]['testY']

                    tempMAE, tempMSE, tempRMSE = fitLinearReg(temp_trainX.loc[:, tempNewSol == 1],
                                                              temp_testX.loc[:, tempNewSol == 1],
                                                              temp_trainY,
                                                              temp_testY)
                    temptempYij.append(-1*tempRMSE)

            tempYij = tempYij + temptempYij
            tempYbari.append(round((1/numReps)*sum(np.array(tempYij)),5))
 
    return i, tempYij, tempYbari 