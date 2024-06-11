# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:41:51 2023

@author: ephouser
"""

###########################
##  Setting Environment  ##
###########################

import csv, math, random, statistics, sys, os, time , pygad
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LassoCV, LinearRegression, Lasso
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support, fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sys.path.insert(0,"/home/ephouser/RS")
sys.path.insert(0,"/home/ephouser/RSP")
sys.path.append('G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\')
from datetime import datetime
from sklearn.metrics import precision_recall_curve
import warnings
from EstimatingPerformance import *
warnings.filterwarnings('ignore')

#####################################
computer = 1 # 0: VCL | 1: Personal #
#####################################

######################
### Experiment paramter
######################
M = 10                  # Marco-Replications
N = 30                  # Undersamples
K = 10                  # post-replication
R = 2                   # Ratio of defects:non-defects (1:p)
R_0 = 54                # Imbalance ratio

gaSeed = 1
np.random.seed(gaSeed) #1
SEED_M = np.random.choice(range(1000), M)

# Set Seeds
random.seed(gaSeed) #777
seedTrainTestMac = random.sample(range(M*10), M)
seedTrainTestMic = random.sample(range(100000), 100000)

def optimalThreshold_Gmeans(yTest, yPredProba):
    
    fpr, tpr, thresholds = roc_curve(yTest, yPredProba)
    gmeans = np.sqrt(tpr * (1-fpr))
    optThreshIndex = gmeans.argmax()
    optThresh = thresholds[optThreshIndex]
    return optThresh


GA_SOL_FITNESS = []

###############################################################################
###############################################################################
###############################################################################

#############################
#############################
##         Phase 1         ##
#############################
#############################

expData = ["DS1","DM1","DL1"]
loadData = ["DS_TEST", "DM_TEST", "DL_TEST"]
trueSol = []

results_TotalTime = pd.DataFrame(data = None)
results_Solutions = pd.DataFrame(data = None, columns = ["Dataset", "Solutions"])
results_Feat_Summary = pd.DataFrame(data = None, columns = ["Experiment", "Dataset","Inf", "Corr", "Noise", "% Correct", "Total Feats"])

#####################
### Dataset-iter  ###
#####################
for EXP_I in range(len(expData)):

    ############################
    ##      Load Dataset      ##
    ############################
    dataName = expData[EXP_I]
    if computer == 0:
        fileName = "/home/ephouser/NP/Data/"+dataName+".csv"        
    elif computer == 1:
        gDrivePath = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\"
        fileName = gDrivePath+dataName+".csv"      

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
    dataX = dataFull.loc[:,dataFull.columns != 'y']

    dataTrainTestMac = {}
    for mi in range(M):
        # Build Train and Test Datasets
        temp_dataX_Train, temp_dataX_Test, temp_dataY_Train, temp_dataY_Test = train_test_split(dataX, dataY, test_size = 0.2, random_state= seedTrainTestMac[mi])
        dataTrainTestMac[mi] = {"X_Train_Mac":temp_dataX_Train, "X_Test_Mac":temp_dataX_Test, 
                                "y_Train_Mac":temp_dataY_Train, "y_Test_Mac":temp_dataY_Test}
    
    trueSol.append([x for x in dataX.columns if 'x' in x])

    ######################
    ### LASSO paramter
    ######################
    alphaGrid = [0.01, 0.1, 1]
    tol = 1e-2
    
    ######################
    ### GA paramter
    ######################
    POP_SIZE = len(dataX.columns)      # GA Parameter: Population Size
    P_CROSSOVER = 0.8            # GA Parameter: Probability of a Crossover
    P_MUTATION = 0.2             # GA Parameter: Probability of a Mutation
    MAX_ITER = len(dataX.columns)     # GA Parameter: Maximum Iterations
    
    ######################
    ### cut off value for feature selection
    ######################
    q = 0.33
    
    EN_threshold = q
    SOEN_threshold = q


    ##################
    ### Macro-iter ###
    ##################
    timeSummary = []
    All_feat_SOEN_OOB = []
    ALL_NEW_SOEN_FEATURES = []
    for i1 in range(M):
        print("--------------------------------------------------------------------------")
        print(f"START MACRO: {i1}/{M-1}")
        
        # Start Run Time
        startTime = datetime.now().strftime("%H:%M:%S")
        startTime = datetime.strptime(startTime, "%H:%M:%S")
        printTime = datetime.strftime(startTime,"%H:%M:%S")
        print(f"START TIME: {printTime}")
        
        dataX_Train_MacI = dataTrainTestMac[i1]["X_Train_Mac"]
        dataY_Train_MacI = dataTrainTestMac[i1]["y_Train_Mac"]
        dataX_Test_MacI = dataTrainTestMac[i1]["X_Test_Mac"]
        dataY_Test_MacI = dataTrainTestMac[i1]["y_Test_Mac"]
        
        # Get Microreplication Data
        dataTrainValidMic = {}
        for ni in range(N):
            # Build Train and Test Datasets
            temp_dataX_Train, temp_dataX_Valid, temp_dataY_Train, temp_dataY_Valid = train_test_split(dataX_Train_MacI, dataY_Train_MacI, test_size = 0.2, random_state= seedTrainTestMic[ni])
            dataTrainValidMic[ni] = {"X_Train_Mic":temp_dataX_Train, "X_Valid_Mic":temp_dataX_Valid, 
                                     "y_Train_Mic":temp_dataY_Train, "y_Valid_Mic":temp_dataY_Valid}
            
        #############
        ### SOEN with OOB
        #############
        def fitness_function(ga_instance, solution, solution_idx):
            
            temp_vars = dataVars[solution == 1]
            temp_scores = []
            
            for n0 in range(N):
                X_Train_Fit = dataTrainValidMic[n0]['X_Train_Mic']
                X_Valid_Fit = dataTrainValidMic[n0]['X_Valid_Mic']
                y_Train_Fit = dataTrainValidMic[n0]['y_Train_Mic']
                y_Valid_Fit = dataTrainValidMic[n0]['y_Valid_Mic']
    
                temp_model = LassoCV(alphas = alphaGrid, tol = tol).fit(X_Train_Fit[temp_vars], y_Train_Fit)                
                coef = pd.Series(temp_model.coef_, index = temp_vars, name = "Sel")
                
                if np.max(np.abs(coef)) == 0:
                    temp_scores.append(-1000000)
                else:
                                        
                    temp_pred = temp_model.predict(X_Valid_Fit[temp_vars])
                    
                    mse = mean_squared_error(y_Valid_Fit, temp_pred)
                    r2 = r2_score(y_Valid_Fit, temp_pred)
                    
                    temp_scores.append(mse)
                    
            return 1 / (np.mean(temp_scores)+0.000001)
            # return -1*np.mean(temp_scores)
                        
        # Define the on_generation callback function to report the current iteration
        def on_generation(ga_instance):
            print("Generation:", ga_instance.generations_completed)
            
        ga_instance = pygad.GA(num_generations = MAX_ITER,
                               sol_per_pop = POP_SIZE,
                               crossover_probability = P_CROSSOVER,
                               mutation_probability = P_MUTATION,
                               num_parents_mating = 10,
                               num_genes=len(dataVars),
                               fitness_func=fitness_function,
                               mutation_type="swap",
                               gene_type=int,
                               random_seed = i1,
                               init_range_low=0,
                               init_range_high=2,
                               on_generation = on_generation)
    
        ga_instance.run()
        
        if computer == 0:
          plotName = f"/home/ephouser/RSP/GA/Plots/GA_LR_{expData[EXP_I]}_{i1+1}.png"
          fig = ga_instance.plot_fitness()
          fig.show()
          fig.savefig(plotName)
          plt.close()
        else:
          ga_instance.plot_fitness()

        
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {(1/solution_fitness)+0.000001}")
              
        GA_SOL_FITNESS.append((1/solution_fitness)+0.000001)
        
        SOEN_var = list(dataVars[ga_instance.best_solution()[0] == 1])
        ALL_NEW_SOEN_FEATURES.append(SOEN_var)
        
        tempALLFEATS_SOEN = []
        for i2 in range(N):
            print(f"START MICRO: {i2}/{N-1}")
            
            temp_dataX_Train = dataTrainValidMic[i2]['X_Train_Mic']
            temp_dataY_Train = dataTrainValidMic[i2]['y_Train_Mic']

            temp_dataX_Valid = dataTrainValidMic[i2]['X_Valid_Mic']    
            temp_dataY_Valid = dataTrainValidMic[i2]['y_Valid_Mic']
            
            # temp_dataX_Test = dataTrainTestMac[i2]['X_Test_Mac']    
            # temp_dataY_Test = dataTrainTestMac[i2]['y_Test_Mac']

            temp_model = LassoCV(alphas = alphaGrid, tol = tol).fit(temp_dataX_Train[SOEN_var], temp_dataY_Train)
                
            coef = pd.Series(temp_model.coef_, index = SOEN_var, name = "Sel")
            
            tempVars = list(coef[coef!=0].index.values)
            tempALLFEATS_SOEN.append(tempVars)
    
        
        All_feat_SOEN_OOB.append(tempALLFEATS_SOEN)
            
        
        endTime = datetime.now().strftime("%H:%M:%S")
        endTime = datetime.strptime(endTime, "%H:%M:%S")    
                
        totalTime = endTime-startTime
        timeSummary.append(round(totalTime.total_seconds() / 60,4))
        
        print(f"TOTAL TIME: {timeSummary[-1]} minutes")

    FINAL_SOEN_FEATS = []
    FEAT_SUMMARY = []
    for i in range(M):
        tempSOEN_FEATS_TO_KEEP = []
        tempFEAT_SUMMARY = []
        for feat in dataVars:
            featCount = 0
            for j in All_feat_SOEN_OOB[i]:
                if feat in j:
                    featCount += 1
            
            if featCount >= N*SOEN_threshold:
                tempSOEN_FEATS_TO_KEEP.append(feat)
                
        xCount = 0
        cCount = 0
        nCount = 0
        for feat in tempSOEN_FEATS_TO_KEEP:
            if 'x' in feat:
                xCount+=1
            elif 'c' in feat:
                cCount +=1
            elif 'n' in feat:
                nCount += 1
        
        FINAL_SOEN_FEATS.append(tempSOEN_FEATS_TO_KEEP)
        FEAT_SUMMARY.append(["GA",expData[EXP_I],xCount, cCount, nCount, round(xCount/len(trueSol[EXP_I]), 3), xCount+cCount+nCount])
    
    
    dataSize = ''.join(char for char in expData[EXP_I] if not char.isnumeric())
            
    totalTimeSummary = pd.DataFrame(timeSummary, columns = [expData[EXP_I]])
    FEAT_SUMMARY = pd.DataFrame(FEAT_SUMMARY, columns = ["Experiment", "Dataset","Inf", "Corr", "Noise", "% Correct", "Total Feats"])
    FINAL_SOEN_FEATS = pd.DataFrame(pd.DataFrame([FINAL_SOEN_FEATS]).values.reshape(M,1), columns = ["Solutions"])
    
    if computer == 0:
        totalTimeSummary.to_csv("/home/ephouser/RSP/GA/Iterations/Summary_Time"+expData[EXP_I]+".csv")
        FEAT_SUMMARY.to_csv("/home/ephouser/RSP/GA/Iterations/Summary_Feats"+expData[EXP_I]+".csv")
        FINAL_SOEN_FEATS.to_csv("/home/ephouser/RSP/GA/Iterations/Summary_Sols"+expData[EXP_I]+".csv")
    else:
        totalTimeSummary.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\totalTimeSummary_"+expData[EXP_I]+".csv")
        FEAT_SUMMARY.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\FEAT_SUMMARY_"+expData[EXP_I]+".csv")
        FINAL_SOEN_FEATS.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\FINAL_SOEN_FEATS_"+expData[EXP_I]+".csv")
    
    
    tempDatasetDF = pd.DataFrame(data = [expData[EXP_I]]*M, columns= ["Dataset"])
    tempFeatDF = pd.concat([tempDatasetDF, FINAL_SOEN_FEATS], axis = 1)
    
    results_TotalTime = pd.concat([results_TotalTime,totalTimeSummary], axis = 1)
    results_Solutions = pd.concat([results_Solutions, tempFeatDF], axis = 0, ignore_index=True)
    results_Feat_Summary = pd.concat([results_Feat_Summary, FEAT_SUMMARY], axis = 0, ignore_index=True)

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

################################
######  Define Functions  ######
################################

# Construct the Train and Test Sets
def getTrainTestData(dataX,dataY,testSize, R, seedSet):
    
    dataTrainTest = {}
    for r in range(R):
        # Build Train and Test Datasets
        trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, random_state= seedSet[r])
        dataTrainTest[r] = {"trainX":trainX, "testX":testX, "trainY":trainY, "testY":testY}
        
    return dataTrainTest

#####################################
##### Phase 2

# LOAD SOLUTIONS
# LOAD DATASETS

# Set Seeds
random.seed(1)
seedSampleNodes = random.sample(range(10000), 1000)
seedSurNodes = random.sample(range(10000), 1000)
seedRepSample = random.sample(range(1,1000000), 999999)
seedIteration = random.sample(range(100000), 100000)
seedTrainTest = random.sample(range(100000), 100000)

# Compile Data
uniqueSize = pd.unique(results_Solutions.iloc[:,0].values)
loadSolutions = pd.DataFrame(data = None, columns = ["Experiment", "Dataset", "Solutions"])
for i in range(len(uniqueSize)):
    tempDF = pd.DataFrame(data = None)
    tempDF["Experiment"] = ["TRUE"] + ["GA"]*len(results_Solutions["Dataset"].loc[results_Solutions["Dataset"] == uniqueSize[i]])    
    tempDF["Dataset"] = [uniqueSize[i]]*(1+len(results_Solutions["Dataset"].loc[results_Solutions["Dataset"] == uniqueSize[i]]))
    tempDF["Solutions"] = [trueSol[i]] + list(results_Solutions["Solutions"].loc[results_Solutions["Dataset"] == uniqueSize[i]])
    
    loadSolutions = pd.concat([loadSolutions, tempDF], axis = 0, ignore_index=True)

# Number of Replications
nREP = 30
Yij = []
Sol_Perf = []

# For Each Dataset Size
for EXP_I in range(len(loadData)):

    ########################################
    ######  Parameter Initialization  ######
    ########################################
    # Start Run Time
    startTime = datetime.now().strftime("%H:%M:%S")
    startTime = datetime.strptime(startTime, "%H:%M:%S")
    
    # Import Data
    datasetName = loadData[EXP_I]
    
    if computer == 0:
        fileName = "/home/ephouser/RSP/Data/"+datasetName+".csv"        
    else:
        # fileName = "C:\\Users\\ethan\\Downloads\\DataXS_1.csv"
        gDrivePath = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\"
        fileName = gDrivePath+loadData[EXP_I]+".csv"   
            
    data = pd.read_csv(fileName)
    dataY = data.loc[:,"y"]                     # Response Data
    dataX = data.loc[:,data.columns != "y"]     # Explanatory Data
    allFeats = dataX.columns
    
    # Shuffle Order of Columns
    shuffle = True
    if shuffle:
        varNames = list(dataX.columns)
        random.seed(7)
        dataX = dataX.loc[:,random.sample(varNames, len(varNames))]
        
    # Construct Train and Test Datasets
    dataTrainTest = getTrainTestData(dataX, dataY, 0.2, nREP, seedTrainTest)
       
    # For Each Solution Corresponding to Dataset Size
    for EXP_J in range(len(loadSolutions)):
                
        if "TRUE" in loadSolutions.iloc[EXP_J,0]:
            expName = "True"
        
        elif "GA" in loadSolutions.iloc[EXP_J,0]:
            expName = "GA"
        
        if loadData[EXP_I][:2] in loadSolutions.iloc[EXP_J, 1]:
            
            tempSol = loadSolutions.iloc[EXP_J, 2]
            tempYbar = []
            tempYij = []
            tempVar = []
            
            if len(tempSol) > 0:
                # For Each Replication
                for EXP_K in range(nREP):
                    
                    trainX = dataTrainTest[EXP_K]['trainX']
                    testX = dataTrainTest[EXP_K]['testX']
                    trainY = dataTrainTest[EXP_K]['trainY']
                    testY  = dataTrainTest[EXP_K]['testY']
                    
                    tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX.loc[:,tempSol],
                                                              testX.loc[:,tempSol],
                                                              trainY,
                                                              testY)
                    tempYij.append(tempRMSE)
                
                tempYij = np.array(tempYij)
                tempYbar = round((1/len(tempYij))*sum(tempYij),5)
                tempVar = round(sum((tempYij-tempYbar)**2)/(nREP-1) ,5)
                
                Sol_Perf.append([expName, loadData[EXP_I][:2], tempYbar, tempVar])
                Yij.append([expName, loadData[EXP_I][:2], tempYij.copy().tolist()])
    
                print(f"{expName} Ybar = {tempYbar}")
                print(f"{expName} Var = {tempVar}")    
                print("\n")

lossDF = pd.DataFrame(data = Sol_Perf, columns = ["Experiment", "Data Size", "Loss", "Variance"])
lossDF["Solution"] = list(loadSolutions.iloc[:,2])

expTypes = lossDF["Experiment"].unique()
dataTypes = lossDF["Data Size"].unique()

expYBAR = np.zeros((len(expTypes),len(dataTypes)))
expStD = np.zeros((len(expTypes),len(dataTypes))) 

for i in range(len(expTypes)):
    for j in range(len(dataTypes)):
        print(f"{expTypes[i]} & {dataTypes[j]} Loss:")
        print(lossDF[(lossDF["Experiment"] == expTypes[i]) & (lossDF["Data Size"] == dataTypes[j])]["Loss"])
        print("Avg Loss: ",round(lossDF[(lossDF["Experiment"] == expTypes[i]) & (lossDF["Data Size"] == dataTypes[j])]["Loss"].mean(),2))
        expYBAR[i][j] = round(lossDF[(lossDF["Experiment"] == expTypes[i]) & (lossDF["Data Size"] == dataTypes[j])]["Loss"].mean(),2)
        expStD[i][j] = round((lossDF[(lossDF["Experiment"] == expTypes[i]) & (lossDF["Data Size"] == dataTypes[j])]["Loss"].std()),2)
        print("\n")

expTypes = np.array(expTypes)

expYBAR = np.vstack((dataTypes, expYBAR))
expStD = np.vstack((dataTypes, expStD))

expTypes = np.insert(expTypes, 0, "Experiment")
expYBAR = np.column_stack((expTypes, expYBAR))
expStD = np.column_stack((expTypes, expStD))

expLOSS = expYBAR[1:, 1:] / expYBAR[1][1:]

for i in range(len(expLOSS)):
    for j in range(len(expLOSS[0])):
        expLOSS[i][j] = round(expLOSS[i][j],2) 

expLOSS = pd.DataFrame(data = expLOSS, columns = ["DS", "DM", "DL"])
expLOSS.insert(0,"Experiment", expTypes[1:])

results_SelectionSummary = pd.DataFrame(data = None, columns = ["Experiment", "Dataset", "Avg % Correct", "ST % Correct", "Avg Total Feats", "SD Total Feats"])
for i in range(len(pd.unique(results_Feat_Summary["Dataset"]))):
    correct_Ybar = round(results_Feat_Summary["% Correct"].loc[results_Feat_Summary["Dataset"] == expData[i]].mean(),3)
    correct_SD = round(results_Feat_Summary["% Correct"].loc[results_Feat_Summary["Dataset"] == expData[i]].std(),3)

    totalFeats_Ybar = round(results_Feat_Summary["Total Feats"].loc[results_Feat_Summary["Dataset"] == expData[i]].mean(),3)
    totalFeats_SD = round(results_Feat_Summary["Total Feats"].loc[results_Feat_Summary["Dataset"] == expData[i]].std(),3)

    tempDF = pd.DataFrame(data = [["GA", pd.unique(results_Feat_Summary["Dataset"])[i], correct_Ybar, correct_SD, totalFeats_Ybar, totalFeats_SD]], columns = results_SelectionSummary.columns)
    results_SelectionSummary = pd.concat([results_SelectionSummary,tempDF], axis = 0, ignore_index = True)

if computer == 0:
    results_TotalTime.to_csv("/home/ephouser/RSP/GA/Summary_TotalTime.csv")
    results_Solutions.to_csv("/home/ephouser/RSP/GA/Summary_Solutions.csv")
    results_Feat_Summary.to_csv("/home/ephouser/RSP/GA/Summary_Features.csv")
    expLOSS.to_csv("/home/ephouser/RSP/GA/Summary_Loss.csv")
    results_SelectionSummary.to_csv("/home/ephouser/RSP/GA/Summary_Selections.csv")     
else:
    results_TotalTime.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\Summary_TotalTime.csv")
    results_Solutions.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\Summary_Solutions.csv")
    results_Feat_Summary.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\Summary_Features.csv")
    expLOSS.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\Summary_Loss.csv")
    results_SelectionSummary.to_csv("G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\GA\\Summary_Selections.csv")




# # CHECK CORRELATION
# corrData = expData
# for corrI in range(len(corrData)):
#     gDrivePath = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\CorrData\\coefcorr_"
#     fileName = gDrivePath+corrData[corrI]+".csv"      
#     data = pd.read_csv(fileName)
    
#     for i in range(len(data)):
#         myNames = []
#         for j in range(len(data.columns)):   
#             if j == 0:
#                 myNames.append("VarName")
#             if j > 0:
                
#                 myNames.append(data.columns[j].replace('coef.', ""))
                
#                 if np.isnan(data.iloc[i,j]):
#                     data.iloc[i,j] = 0.0
#                 else:
#                     data.iloc[i,j] = 1.0
    
#     data.columns = myNames
#     varName = "CORR_" + corrData[corrI]
    
#     exec(varName + " = data")
