# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:41:07 2024

@author: ephouser
"""

from PythonModules import *

# Construct the Train and Test Sets
def getTrainTestData(dataX, dataY, testSize, R, seedSet):
    
    dataTrainTest = {}
    for r in range(R):
        # Build Train and Test Datasets
        trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, random_state= seedSet[r])
        dataTrainTest[r] = {"trainX":trainX, "testX":testX, "trainY":trainY, "testY":testY}
        
    return dataTrainTest

# Add Additional Train and Test Sets As Needed
def getMoreTrainTestData(dataX, dataY, testSize, R, seedSet, dataTrainTest):
    prevR = len(dataTrainTest)
    for r in range(R):
        # Build Train and Test Datasets
        trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, random_state= seedSet[r])
        dataTrainTest[r+prevR] = {"trainX":trainX, "testX":testX, "trainY":trainY, "testY":testY}
        
    return dataTrainTest
