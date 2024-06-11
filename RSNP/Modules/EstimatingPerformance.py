# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:14:19 2023

@author: Ethan Houser

"""

from PythonModules import *

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
    RMSE = np.sqrt(metrics.mean_squared_error(testY, yPred))    #Root Mean Squared Error
    
    
    # ########################
    # # Calculate AIC and BIC
    # import statsmodels.api as sm

    # ###############
    # # Fit the Model
    # X_train = sm.add_constant(trainX)
    # model = sm.OLS(trainY, X_train).fit()
    
    # ##################################
    # # Compute Likelihood for Test Data
    # X_test = sm.add_constant(testX)
    # y_test_pred = model.predict(X_test)
    # residuals = testY - y_test_pred
    # sigma_squared = np.mean(residuals ** 2)
    # likelihood = np.prod(1 / np.sqrt(2 * np.pi * sigma_squared) * np.exp(-residuals ** 2 / (2 * sigma_squared)))
    
    # # Get the number of parameters in the model
    # num_params = len(model.params)
    
    # # Compute AIC and BIC
    # n = len(testY)
    # aic = 2 * num_params - 2 * np.log(likelihood)
    # bic = np.log(n) * num_params - 2 * np.log(likelihood)
    
    # print("AIC on testing data:", aic)
    # print("BIC on testing data:", bic)
    
    
    return MAE, MSE, RMSE

def optimalThreshold_Gmeans(yTest, yPredProba):
    fpr, tpr, thresholds = roc_curve(yTest, yPredProba)  ## Implement All vs One Strategy
    gmeans = np.sqrt(tpr * (1-fpr))
    optThreshIndex = gmeans.argmax()
    optThresh = thresholds[optThreshIndex]
    return optThresh

def fitLogReg(trainX, testX, trainY, testY):
    
    if len(np.unique(trainY))==2:
        model = LogisticRegression()
    elif len(np.unique(trainY))>2:
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    
    # Build Training Model
  
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

