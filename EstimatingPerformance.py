# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:14:19 2023

@author: ethan
"""

import numpy as np
import pandas as pd
import math, random, statistics
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support, fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

##########################
## Experiment Parameters
##########################

M = 30                      # Replications
trainSize = 0.8             # Size of Training Set
testSize = 1- trainSize     # Size of Testing Set

random.seed(123)
seedsM = random.sample(range(1000), M)      #Create CRN stream

modelType = 1       #0 = Linear Regression, 1 = Logistic Regression

##########################
## Defining Functions
##########################

#############################
### Linear Regression Model

def fitLinearReg(trainX, testX, trainY, testY):
    
    # Build Training Model
    model = LinearRegression()    
    model.fit(trainX, trainY)
    
    # Make Predictions
    yPred = model.predict(testX)
    yPred = [round(i) for i in yPred]
    
    # Want values to be as small as possible
    MAE = metrics.mean_absolute_error(testY, yPred)             #Mean Absolute Error
    MSE = metrics.mean_squared_error(testY, yPred)              #Mean Squared Error
    RMSE = np.sqrt(metrics.mean_squared_error(testY, yPred))    #Roo Mean Squared Error
    return MAE, MSE, RMSE

def optimalThreshold_Gmeans(yTest, yPredProba):
    fpr, tpr, thresholds = roc_curve(yTest, yPredProba)
    gmeans = np.sqrt(tpr * (1-fpr))
    optThreshIndex = gmeans.argmax()
    optThresh = thresholds[optThreshIndex]
    return optThresh

def fitLogReg(trainX, testX, trainY, testY):
    
    # Build Training Model
    model = LogisticRegression()    
    model.fit(trainX, trainY)
    
    # Make Predictions
    trainPred = model.predict(trainX)
    opt_threshold = optimalThreshold_Gmeans(trainY, trainPred)
    temp_pred = model.predict(testX)
    temp_pred = list(map(bool, temp_pred >= opt_threshold))
    tn, fp, fn, tp = confusion_matrix(testY, temp_pred).ravel()

    
    precision = precision_score(testY, temp_pred)
    recall = recall_score(testY, temp_pred)
    f1_score = fbeta_score(testY, temp_pred, beta = 1)
    f2_score = fbeta_score(testY, temp_pred, beta = 2)
    f3_score = fbeta_score(testY, temp_pred, beta = 3)
    f4_score = fbeta_score(testY, temp_pred, beta = 4)
    
    return f1_score, f2_score, f3_score, f4_score, precision, recall
    
    # Build Training Model
    model = LinearRegression()    
    model.fit(trainX, trainY)
    
    yPred = model.predict(testX)
    yPred = [round(i) for i in yPred]
    
    # Want values to be as small as possible
    MAE = metrics.mean_absolute_error(testY, yPred)             #Mean Absolute Error
    MSE = metrics.mean_squared_error(testY, yPred)              #Mean Squared Error
    RMSE = np.sqrt(metrics.mean_squared_error(testY, yPred))    #Roo Mean Squared Error
    return MAE, MSE, RMSE

#Load the data
#data = pd.read_excel("C:\\Users\\ethan\\Downloads\\ScreeningData_LR.xlsx")
data = pd.read_excel("C:\\Users\\ethan\\Downloads\\ScreeningData_LogReg.xlsx")

#Create X and Y Datasets
dataX = data.iloc[:,1:]
dataY = data["y"]

all_result = []

#List of Variables
varvar = list(range(0,10))

colNames = ["Statistic"]+[str("Itr "+str(i+1)) for i in range(M)]

if modelType == 0:
    results = pd.DataFrame(data=None, columns=colNames)
    results.iloc[:,0] = ["MAE", "MSE", "RMSE"]
elif modelType == 1:
    results = pd.DataFrame(data=None, columns=colNames)
    results.iloc[:,0] = ["F1 Score", "F2 Score", "F3 Score", "F4 Score", "Precision", "Recall"]


m=0
for m in range(M):
    
    # Build Train and Test Datasets
    trainX, testX, trainY, testY = train_test_split(dataX.iloc[:,varvar], dataY, test_size = 0.2, random_state=seedsM[m])
    
    if modelType == 0:      # Linear Regression Model
    
        tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX, testX, trainY, testY)
        
        results.iloc[0,m+1] = tempMAE
        results.iloc[1,m+1] = tempMSE
        results.iloc[2,m+1] = tempRMSE
        
    elif modelType == 1:    # Logistic Regression Model
        f1_score, f2_score, f3_score, f4_score, precision, recall = fitLogReg(trainX, testX, trainY, testY)
        tempData = [f1_score, f2_score, f3_score, f4_score, precision, recall]
        results.iloc[:, m+1] = tempData
        

##########################
## Averaging Results
##########################

if modelType == 0:    
    print("Average MAE:", round(statistics.mean(results.iloc[0,1:]), 3))
    print("Average MSE:", round(statistics.mean(results.iloc[1,1:]), 3))
    print("Average RMSE:", round(statistics.mean(results.iloc[2,1:]), 3))    
elif modelType == 1:
    print("Average F1 Score", round(statistics.mean(results.iloc[0,1:]),3))
    print("Average F2 Score", round(statistics.mean(results.iloc[1,1:]),3))
    print("Average F3 Score", round(statistics.mean(results.iloc[2,1:]),3))
    print("Average F4 Score", round(statistics.mean(results.iloc[3,1:]),3))
    print("Average Precision", round(statistics.mean(results.iloc[4,1:]),3))
    print("Average Recall", round(statistics.mean(results.iloc[5,1:]),3))











