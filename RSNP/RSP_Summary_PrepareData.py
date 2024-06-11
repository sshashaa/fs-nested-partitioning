# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:57:00 2024

@author: ephouser
"""

import numpy as np
import pandas as pd
import math
from scipy.stats import t

def prepareData(finalP2RESULTS):
    
    ###############################################################################
    ###############################################################################
    ##### CREATE RESULTS SUMMARY
    #####

    expTypes = finalP2RESULTS["Model"].unique()
    dataTypes = finalP2RESULTS["Data Size"].unique()

    # Time Summary
    expTimeAvg = np.zeros((len(expTypes),len(dataTypes)))
    expTimeSD = np.zeros((len(expTypes),len(dataTypes))) 

    # Percent Correct
    expPercentCorrectAvg = np.zeros((len(expTypes),len(dataTypes)))
    expPercentCorrectSD = np.zeros((len(expTypes),len(dataTypes))) 

    # Total Feats
    expTotalFeatsAvg = np.zeros((len(expTypes),len(dataTypes)))
    expTotalFeatsSD = np.zeros((len(expTypes),len(dataTypes))) 
    expTotalFeats_Quant5 = np.zeros((len(expTypes),len(dataTypes))) 
    expTotalFeats_Quant95 = np.zeros((len(expTypes),len(dataTypes))) 

    # Loss
    expLossAvg = np.zeros((len(expTypes),len(dataTypes)))
    expLossSD = np.zeros((len(expTypes),len(dataTypes))) 

    # Percentage Loss
    expPerLoss = np.zeros((len(expTypes),len(dataTypes))) 

    # Summary
    expSummary = np.zeros((len(expTypes),len(dataTypes))).astype('str')


    for i in range(len(expTypes)):
        for j in range(len(dataTypes)):

            # Time
            expTimeAvg[i][j] = round(finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Time"].mean(),2)
            expTimeSD[i][j] = round((finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Time"].std()),2)        
            
            # Percent Correct
            expPercentCorrectAvg[i][j] = round(finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Percent Correct"].mean(),2)
            expPercentCorrectSD[i][j] = round((finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Percent Correct"].std()),2)        
            
            # Total Features
            expTotalFeatsAvg[i][j] = round(finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Total Feats"].mean(),2)
            expTotalFeatsSD[i][j] = round((finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Total Feats"].std()),2)        
            expTotalFeats_Quant5[i][j] = finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Total Feats"].quantile(0.05)
            expTotalFeats_Quant95[i][j] = finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Total Feats"].quantile(0.95)
            
            # Loss
            expLossAvg[i][j] = round(finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Loss"].mean(),2)
            expLossSD[i][j] = round((finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Loss"].std()),2)        
            if math.isnan(expLossSD[i][j]):
                expLossSD[i][j] = 0
            
            # Percentage Loss
            expPerLoss[i][j] = round((finalP2RESULTS[(finalP2RESULTS["Model"] == expTypes[i]) & (finalP2RESULTS["Data Size"] == dataTypes[j])]["Percent Loss"].mean()),2)
                    
            # Summary
            expSummary[i][j] = str(expLossAvg[i][j])+' +/- '+str(expLossSD[i][j])

    # Format Dataframes
    expTypes = np.array(expTypes)

    expLossAvg = np.vstack((dataTypes, expLossAvg))
    expLossSD = np.vstack((dataTypes, expLossSD))
    expSummary = np.vstack((dataTypes, expSummary))
    expPerLoss = np.vstack((dataTypes, expPerLoss))

    expTypes = np.insert(expTypes, 0, "Experiment")
    expLossAvg = np.column_stack((expTypes, expLossAvg))
    expLossSD = np.column_stack((expTypes, expLossSD))

    expSummary = np.column_stack((expTypes, expSummary))
    expPerLoss = np.column_stack((expTypes, expPerLoss))

    # Prepare Data for Export
    expSummary = pd.DataFrame(data = expSummary[1:][:], columns = expSummary[0])
    expPerLoss = pd.DataFrame(data = expPerLoss[1:][:], columns = expPerLoss[0])

    #######################
    ## Compile Data By Size

    colNames = ['Model', 'Data Size', 'Feat Size', 
                #------------------------------------------------------------------------------------------
                # Average & Variance
                'Loss Avg',            'Loss Var',            'Loss 95% HW',
                'Percent Loss Avg',    'Percent Loss Var',    'Percent Loss 95% HW',
                'Time Avg',            'Time Var',            'Time 95% HW',
                'Replications Avg',    'Replications Var',    'Replications 95% HW',
                'Percent Correct Avg', 'Percent Correct Var', 'Percent Correct 95% HW',
                'Total Feats Avg',     'Total Feats Var',     'Total Feats 95% HW',
                #------------------------------------------------------------------------------------------
                # 10%, 50%, 90% Quantile
                'Loss 10%Q',            'Loss 50%Q',            'Loss 90%Q',
                'Percent Loss 10%Q',    'Percent Loss 50%Q',    'Percent Loss 90%Q', 
                'Time 10%Q',            'Time 50%Q',            'Time 90%Q', 
                'Replications 10%Q',    'Replications 50%Q',    'Replications 90%Q',
                'Percent Correct 10%Q', 'Percent Correct 50%Q', 'Percent Correct 90%Q', 
                'Total Feats 10%Q',     'Total Feats 50%Q',     'Total Feats 90%Q'
                ]
    
    finalResults = pd.DataFrame(data = None, columns= colNames)
    uniqueModels = sorted(finalP2RESULTS[finalP2RESULTS['Model'] != 'TRUE']['Model'].unique())

    # MODEL ORDER
    uniqueModels = ['GA', 'NP-Min-SIZ', 'ORS1-Shrinking', 'ORS2-Shrinking', 'ORS3-Shrinking', 'ORS20-Shrinking',   
                   'RS1P-Hybrid-Min-SIZ', 'RS1P-Shrinking-Min-SIZ',
                   'RS2P-Hybrid-Min-SIZ', 'RS2P-Shrinking-Min-SIZ', 
                   'RS3P-Hybrid-Min-SIZ', 'RS3P-Shrinking-Min-SIZ',
                   'RS20P-Hybrid-Min-SIZ', 'RS20P-Shrinking-Min-SIZ',
                   'RS20-Hybrid_n1','RS20-Hybrid_n2', 'RS20-Hybrid_n3', 'RS20-Hybrid_n4',
                   'RS20P-Hybrid_n1-Min-SIZ','RS20P-Hybrid_n2-Min-SIZ', 'RS20P-Hybrid_n3-Min-SIZ', 'RS20P-Hybrid_n4-Min-SIZ', 'RS20P-Hybrid_n5-Min-SIZ']
    
    # FOR EACH MODEL
    for modelName in uniqueModels:
        
        # FILTER DATA
        dataFilteredByModel = finalP2RESULTS[finalP2RESULTS['Model'] == modelName]
        uniqueDataSizes = dataFilteredByModel['Data Size'].unique()
        
        # FOR EACH DATA SIZE
        for dataSize in uniqueDataSizes:
            
            # FILTER DATA
            dataFilteredBySize = dataFilteredByModel[dataFilteredByModel['Data Size'] == dataSize]
            
            if dataSize == 'S':
                tempFeatSize = 15
            elif dataSize == 'M':
                tempFeatSize = 60
            elif dataSize == 'L':
                tempFeatSize = 150
            else:
                tempFeatSize = 250 
            
            numReps = len(dataFilteredBySize)
            numRepsDF = numReps - 1
            
            ######
            # LOSS
            
            # breakpoint()
            
            lossAvg = round(dataFilteredBySize['Loss'].mean(),3)
            lossVar = round(dataFilteredBySize['Loss'].var(),3)
            lossHW  = t.ppf(0.975, df=numRepsDF) * math.sqrt(lossVar / numReps)
            loss10Q = np.quantile(dataFilteredBySize['Loss'], 0.1)
            loss50Q = np.quantile(dataFilteredBySize['Loss'], 0.5)
            loss90Q = np.quantile(dataFilteredBySize['Loss'], 0.9)
            
            ##############
            # PERCENT LOSS
            perLossAvg = round(dataFilteredBySize['Percent Loss'].mean(),3)
            perLossVar = round(dataFilteredBySize['Percent Loss'].var(),3)
            perLossHW  = t.ppf(0.975, df=numRepsDF) * math.sqrt(perLossVar / numReps)
            perLoss10Q = np.quantile(dataFilteredBySize['Percent Loss'], 0.1)
            perLoss50Q = np.quantile(dataFilteredBySize['Percent Loss'], 0.5)
            perLoss90Q = np.quantile(dataFilteredBySize['Percent Loss'], 0.9)
            
            ######
            # TIME
            timeAvg = round(dataFilteredBySize['Time'].mean(),3)
            timeVar = round(dataFilteredBySize['Time'].var(),3)
            timeHW  = t.ppf(0.975, df=numRepsDF) * math.sqrt(timeVar / numReps)
            time10Q = np.quantile(dataFilteredBySize['Time'], 0.1)
            time50Q = np.quantile(dataFilteredBySize['Time'], 0.5)
            time90Q = np.quantile(dataFilteredBySize['Time'], 0.9)
            
            ##############
            # REPLICATIONS
            repsAvg = round(dataFilteredBySize['Replications'].mean(),3)
            repsVar = round(dataFilteredBySize['Replications'].var(),3)
            repsHW  = t.ppf(0.975, df=numRepsDF) * math.sqrt(repsVar / numReps)
            reps10Q = np.quantile(dataFilteredBySize['Replications'], 0.1)
            reps50Q = np.quantile(dataFilteredBySize['Replications'], 0.5)
            reps90Q = np.quantile(dataFilteredBySize['Replications'], 0.9)
            
            #################
            # PERCENT CORRECT
            perCorAvg = round(dataFilteredBySize['Percent Correct'].mean(),3)
            perCorVar = round(dataFilteredBySize['Percent Correct'].var(),3)
            perCorHW  = t.ppf(0.975, df=numRepsDF) * math.sqrt(perCorVar / numReps)
            perCor10Q = np.quantile(dataFilteredBySize['Percent Correct'], 0.1)
            perCor50Q = np.quantile(dataFilteredBySize['Percent Correct'], 0.5)
            perCor90Q = np.quantile(dataFilteredBySize['Percent Correct'], 0.9)
            
            ################
            # TOTAL FEATURES
            totalFeatsAvg = round(dataFilteredBySize['Total Feats'].mean(),3)
            totalFeatsVar = round(dataFilteredBySize['Total Feats'].var(),3)
            totalFeatsHW  = t.ppf(0.975, df=numRepsDF) * math.sqrt(totalFeatsVar / numReps)
            totalFeats10Q = np.quantile(dataFilteredBySize['Total Feats'], 0.1)
            totalFeats50Q = np.quantile(dataFilteredBySize['Total Feats'], 0.5)
            totalFeats90Q = np.quantile(dataFilteredBySize['Total Feats'], 0.9)
            
            ##############
            # COMPILE DATA

            tempData = [modelName, dataSize, tempFeatSize, 
                        #------------------------------------------------------------------------------------------
                        # Average & Variance
                        lossAvg,       lossVar,       lossHW,
                        perLossAvg,    perLossVar,    perLossHW,    
                        timeAvg,       timeVar,       timeHW,       
                        repsAvg,       repsVar,       repsHW,       
                        perCorAvg,     perCorVar,     perCorHW,     
                        totalFeatsAvg, totalFeatsVar, totalFeatsHW, 
                        #------------------------------------------------------------------------------------------
                        # 10%, 50%, 90% Quantile
                        loss10Q,       loss50Q,       loss90Q,
                        perLoss10Q,    perLoss50Q,    perLoss90Q,
                        time10Q,       time50Q,       time90Q,
                        reps10Q,       reps50Q,       reps90Q,
                        perCor10Q,     perCor50Q,     perCor90Q, 
                        totalFeats10Q, totalFeats50Q, totalFeats90Q
                        ]
            
            tempDF = pd.DataFrame(data = [tempData], columns = finalResults.columns)
            
            finalResults = pd.concat([finalResults,tempDF], axis = 0, ignore_index = True)
    

    return finalResults



















