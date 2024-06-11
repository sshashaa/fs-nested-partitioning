# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:53:09 2024

@author: ephouser
"""

#################################################
#################################################
#### Define User Settings

#####################################
computer = 0 # 0: VCL | 1: Personal #
#####################################

#####################################
INDIV = 0 # 0: NO | 1: YES
#####################################

#####################################
if INDIV == 0:
    MACRO_J = list(range(10))
else:
    MACRO_J = [0] # Macro run
#####################################

#################################################
#################################################
#### Import Modules

# Setting Path for Modules
import sys
if computer == 0:
    sys.path.insert(0,"/home/ephouser/RSP/Modules/")
else:
    sys.path.append('G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\Ethan\\Code\\RSP\\Modules\\')

from PythonModules import *                   # General Python Modules (numpy, pandas, etc..)
from LoadDataset import *                     # Module for Importing and Formating Datasets
from EstimatingPerformance import *           # Module for Measuring Performance of Solutions
from ParallelProcessing import *              # Module for Implementing Parallel Processing
from GeneratingDatasets import *              # Module for Generating Macro/Micro Datasets
from RapidScreening import *                  # Module for Rapid Screening
from NestedPartitioning import *              # Module for Nested Partitioning
from Phase2Results import *                   # Module for Computing the Phase 2 Results (Results of Final Solutions)


#################################################
#################################################
#### Define Experiment Settings

expI_Data = ["DS1"]                                         # Dataset
expJ_Macros = MACRO_J                                       # Number of Macroreplications
expJ_Micros = 100                                           # Number of Microreplications
expK_Screen = [1]                                         # 0: No RS | 1: RS 
expL_Screen = ["Hybrid"]                                    # Search Type
expM_Partition = [1]                                        # 0: No NP | 1: NP
expN_SettingPerf = [1]                                      # 0: Avg | 1: Min | 2: Max
expO_SettingIZ = [0,1]                                       # 0: Static | 1: Dynamic | 2: Dynamic BT

myR = list(range(5,int(math.ceil(50 / 5.0)) * 5,5))         # Tracking Rapid Screening Progress
RS_Q_Percent = 0.7                                          # Percent of Features to Screen Out
RS_P_Percent = 0.05                                         # Percent of Features to Automatically Keeo                                            
nREP = 30

#################################################
#################################################
#### Create Containers

# General
colNames = ["Dataset","Macro","R","Search", "Solution", "Time"]                                                 # Reused Column Names
trueSol = []                                                                                                    # List of True Features

# Nested Partitioning
backtrackSummary = pd.DataFrame(data=None)                                                                      # Backtracking Summary
time_ALL_EXP = pd.DataFrame(data=None, columns = ["Experiment" ,"Total Time"])                                  # Total Time for each Experiment
finalSolDict = {}                                                                                               # Dictionary of Final Solutions
finalSelDict = {}                                                                                               # Dictionary of Final Feature Selections
finalSolSelectionDF = pd.DataFrame(data = None)                                                                 # DataFrame of Final Feature Selections ??
finalSolsDF = pd.DataFrame(data = None)                                                                         # DataFrame of Final Solutions ??
finalPartSolSummaryDF = pd.DataFrame(data= ["Inf", "Cor", "Noise"], columns=["Type"])                           # DataFrame of Final Partitioning Summary

#################################################
#################################################
#### Define Seeds

# Set Seeds
random.seed(1)
seedTrainTestMac = random.sample(range(100000), 100000)
seedTrainTestMic = random.sample(range(100000), 100000)

# Phase 1
seedSampleNodes = random.sample(range(10000), 1000)
seedSurNodes = random.sample(range(10000), 1000)
seedRepSample = random.sample(range(1,1000000), 999999)
seedIteration = random.sample(range(100000), 100000)

inputsSeeds = [seedSampleNodes, seedSurNodes, seedRepSample, seedIteration]

# Phase 2
seedTrainTest = random.sample(range(100000), 100000)

#################################################
#################################################
#### Run Experiment

##################
# For Each Dataset
for EXP_I in range(len(expI_Data)):
    
    ########################
    ####  Load Dataset  ####
    ########################
    datasetName = expI_Data[EXP_I]
    dataFull, dataY, dataX_All, allFeats, trueSol, dataTrainTestMac = loadData(datasetName, computer, expJ_Macros, seedTrainTestMic)
    
    ########################################
    ######  Parameter Initialization  ######
    ########################################
    
    ###################              
    # MACROREPLICATIONS    
    for EXP_J in expJ_Macros:
        print(" ")
        print("--------------------------------------------------------------------------")
        print(f"START MACRO: {EXP_J+1}/{max(expJ_Macros)+1}")
        
        #####################
        # Initialize Counters
        k = 0                   # Tree Level
        featCount = 0           # Current Feature
        countNode = 0           # Current Node
        iSS = 0                 # Current Sample Count
        iSR = 0                 # Current Replication Count
        
        ##########################
        # Turn EXP_J into a String
        EXP_J_STR = "0" + str(EXP_J+1) if (EXP_J+1) < 10 else str(EXP_J+1)
                            
        #######################################
        # Parse the X and Y Train and Test Sets
        dataX_Train_MacJ = dataTrainTestMac[EXP_J]["trainX"]
        dataY_Train_MacJ = dataTrainTestMac[EXP_J]["trainY"]
        dataX_Test_MacJ = dataTrainTestMac[EXP_J]["testX"]
        dataY_Test_MacJ = dataTrainTestMac[EXP_J]["testY"]
        
        ##################################
        # Compile X and Y Data For Storage
        dataX_MacJ = [dataX_Train_MacJ, dataX_Test_MacJ]
        dataY_MacJ = [dataY_Train_MacJ, dataY_Test_MacJ]
        
        #######################################
        # Get Microreplication Data for MACRO J
        dataTrainValidMic = {}
        for Ji in range(expJ_Micros):
            # Build Train and Test Datasets
            temp_dataX_Train, temp_dataX_Valid, temp_dataY_Train, temp_dataY_Valid = train_test_split(dataX_Train_MacJ, dataY_Train_MacJ, test_size = 0.2, random_state= seedTrainTestMic[Ji])
            dataTrainValidMic[Ji] = {"trainX":temp_dataX_Train, "testX":temp_dataX_Valid, 
                                     "trainY":temp_dataY_Train, "testY":temp_dataY_Valid}
        
        #####################
        # SCREEN vs NO SCREEN
        for EXP_K in expK_Screen:
            
            ################
            # Compile Inputs
            inputsExpSettings = [computer, INDIV, datasetName, EXP_I, EXP_J, EXP_J_STR, EXP_K, expL_Screen]
            inputsData = [trueSol, dataX_All, dataX_MacJ, dataY_MacJ, dataTrainValidMic]
            inputsParameters = [RS_Q_Percent, RS_P_Percent, iSR, seedTrainTestMic]
            
            ###########################
            # Implement Rapid Screening
            resultsScreen = prepareScreening(inputsExpSettings, inputsData, inputsParameters)

            ##########################
            # Unpack Screening Results
            inputsExpSettings, repScreening, screenTime, screenTimeMin, keptFeats, npOrder, iSR, dataTrainValidMic, dataX_All, dataX_Filtered = resultsScreen
            computer, INDIV, datasetName, EXP_I, EXP_J, EXP_J_STR, EXP_K, EXP_L, expL_Screen = inputsExpSettings
            dataX, dataX_Train_MacJ, dataX_Test_MacJ = dataX_Filtered
            dataX_MacJ = [dataX_Train_MacJ, dataX_Test_MacJ]
            
            ############################
            # PARTITION VS NO PARTITIONG
            for EXP_M in expM_Partition:
                
                ################
                # Compile Inputs
                inputsExpSettings = [computer, INDIV, datasetName, EXP_I, EXP_J, EXP_J_STR, EXP_K, EXP_L, expL_Screen]
                inputsData = [trueSol, dataX_All, dataX, dataY, dataX_MacJ, dataY_MacJ, dataTrainValidMic]
                inputsParameters = [iSR, seedTrainTestMic]
                inputsContainers = [time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]
                inputsScreenResults = [repScreening, screenTime, screenTimeMin, keptFeats, npOrder, iSR, dataTrainValidMic]
                inputsPartitioning = [EXP_M, expN_SettingPerf, expO_SettingIZ]
                
                ###############################
                # Implement Nested Partitioning
                resultsPartition = preparePartitioning(inputsExpSettings, inputsData, inputsParameters, inputsContainers, inputsScreenResults, inputsPartitioning, inputsSeeds)
                


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#######################
#######################
#####   PHASE 2   #####
#######################
#######################

print(" ")
print("--------------------------------------------------------------------------")
print(f"START PHASE 2")

################
# IMPORT RESULTS
if computer == 0:
    resultsSummary_Path = f"/home/ephouser/RSP/Results/AllResultsSummary.xlsx"
else:
    resultsSummary_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\AllResultsSummary.xlsx"

resultsSummary = pd.read_excel(resultsSummary_Path, index_col=None)

P2Data = ["DS_TEST","DM_TEST","DL_TEST","DXL_TEST"]          # Data for Phase 2 Testing

#################
# FOR EACH RESULT
P2_Ybar = []
P2_Yij = []
P2_Var = []

for i in range(len(P2Data)):

    datasetName = P2Data[i]
    mySize = datasetName.replace("_TEST", "")
    
    filteredResultsSummary = resultsSummary[resultsSummary['Data Instance'].str.contains(mySize[1])]
    filteredResultsSummary["Data Size"] = mySize
    
    for j in range(len(filteredResultsSummary)):
            
        modelName = filteredResultsSummary.iloc[j,0]
        dataInstance = filteredResultsSummary.iloc[j,1]
        expSol = ast.literal_eval(filteredResultsSummary.iloc[j,2])
        
        ##############
        # TEST RESULTS
        tempYij, tempYbar, tempVar, allFeats_P2 = getP2Results(computer, datasetName, expSol, nREP, seedTrainTest)
        
        #############
        # STORE TESTS
        P2_Ybar.append([modelName, dataInstance, tempYbar])
        P2_Var.append([modelName, dataInstance, tempVar])
        P2_Yij.append([modelName, dataInstance, tempYij])
        
        ################
        # GET TRUE FEATS
        trueSol = [x for x in allFeats_P2 if 'x' in x]
        
        #################
        ### GET TRUE LOSS
        tempYij, trueLoss, tempVar, allFeats_P2 = getP2Results(computer, datasetName, trueSol, nREP, seedTrainTest)
        
        ########################
        # CALCULATE PERCENT LOSS
        tempPercentLoss = round(tempYbar / trueLoss,3)
        
        tempFinalResults = pd.DataFrame(data = [[modelName, dataInstance, expSol, tempYbar, tempVar, tempPercentLoss, mySize]], columns = ['Model', 'Data Instance', 'Solution', 'Loss', 'Variance', 'Percent Loss', 'Data Size'])
         
        ###########################
        ###  EXPORT TESTING RESULTS
        
        # Prepare File Path
        if computer == 0:
            finalP2RESULTS_Path = f"/home/ephouser/RSP/Results/Phase2Results.xlsx"
        else:
            finalP2RESULTS_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\Phase2Results.xlsx"
    
        # Import Results (If Existing)
        if os.path.isfile(finalP2RESULTS_Path):
            finalP2RESULTS = pd.read_excel(finalP2RESULTS_Path, index_col=None)
        else:
            finalP2RESULTS = pd.DataFrame(data = None, columns = ["Model", "Data Instance", "Solution", "Loss", "Variance", "Percent Loss", "Data Size"])
                        
        # Add/Update Results DF
        if len(finalP2RESULTS) == 0:
            # Add row to the end of the dataframe
            finalP2RESULTS = pd.concat([finalP2RESULTS,tempFinalResults], axis = 0, ignore_index=True)
            
        elif ((finalP2RESULTS['Model'] == tempFinalResults.iloc[0,0]) & (finalP2RESULTS["Data Instance"] == tempFinalResults.iloc[0,1])).any():
                           
            # Replace the existing row
            finalP2RESULTS.loc[(finalP2RESULTS['Model'] == tempFinalResults.iloc[0,0]) & (finalP2RESULTS['Data Instance'] == tempFinalResults.iloc[0,1])] = tempFinalResults
            
        else:
            finalP2RESULTS = pd.concat([finalP2RESULTS,tempFinalResults], axis = 0, ignore_index=True)
            
        # Export Results
        finalP2RESULTS.to_excel(finalP2RESULTS_Path, index = False)
          

print("Done!")





