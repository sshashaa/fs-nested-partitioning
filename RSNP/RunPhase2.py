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
#### Define Inputs

# Phase 2
nREP = 100
seedTrainTest = random.sample(range(100000), 100000)

#################################################
#################################################
#### IMPORT RESULTS
if computer == 0:
    resultsSummary_Path = f"/home/ephouser/RSP/Results/AllResultsSummary.xlsx"
else:
    resultsSummary_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\AllResultsSummary.xlsx"

resultsSummary = pd.read_excel(resultsSummary_Path, index_col=None)
resultsSummary['Data Size'] = [dataInstance.strip('0123456789') for dataInstance in resultsSummary['Data Instance']]
resultsSummary = resultsSummary.sort_values(by = ["Data Instance"]).reset_index(drop=True)

#######################
#######################
#####   PHASE 1   #####
#######################
#######################

############################
# BUILD RS CONVERGENCE PLOTS

if computer == 0:
    allConvData_Path = f"/home/ephouser/RSP/Results/SideExp/"
else:
    allConvData_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\"
 
# Filter Data
uniqueModels = resultsSummary[["Model", "Data Size"]].drop_duplicates()
uniqueModels = uniqueModels[uniqueModels['Model'].str.contains("RS-")].reset_index(drop=True)

for i in range(len(uniqueModels)):
    
    # Initialize the plot
    plt.figure()
   
    modelName = uniqueModels['Model'].iloc[i] 
    searchType = modelName.split("-")[-1]
    dataSize = uniqueModels['Data Size'].iloc[i]  
    
    if "X" not in dataSize:
    
        allConvData_FullPath = allConvData_Path + "AllConvPlotData_" + dataSize + "-" + searchType + ".xlsx"
        convPlotData = pd.read_excel(allConvData_FullPath, index_col=None)
        
        for dI in convPlotData['Data Instance'].unique():
            
            convPlotData_Filtered = convPlotData[convPlotData['Data Instance'] == dI].reset_index(drop=True)
            maxItr = max(convPlotData_Filtered["Itr"])
            
            # Plot a new line for each iteration
            plt.plot(convPlotData_Filtered["Itr"], convPlotData_Filtered["Mean"], marker='o', label=f'MACRO-{dI} (Max Itr = {maxItr})')
   
        # Add labels and legend
        plt.title(f"{modelName}-{dataSize}")
        plt.xlabel('Iteration #')
        plt.xlim(0, 20)
        plt.xticks(range(0, 21, 2))
        plt.ylabel('Mean')
        plt.legend()
        
        # Save the plot as a PNG file after each iteration
        if computer == 0:
            plt.savefig(f'/home/ephouser/RSP/Results/SideExp/ConvergencePlot_{modelName}_{dataSize}.png')
            plt.close()  
        else:
            plt.savefig(f'G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\ConvergencePlot_{searchType}_{EXP_I}_{EXP_J+1}.png')
            plt.close()


#######################
#######################
#####   PHASE 2   #####
#######################
#######################

print(" ")
print("--------------------------------------------------------------------------")
print(f"START PHASE 2")

P2Data = ["DS_TEST","DM_TEST","DL_TEST"]          # Data for Phase 2 Testing
# P2Data = ["DS_TEST","DM_TEST","DL_TEST","DXL_TEST"]          # Data for Phase 2 Testing
# P2Data = ["DL_TEST","DXL_TEST"]          # Data for Phase 2 Testing

#################
# FOR EACH RESULT
P2_Ybar = []
P2_Yij = []
P2_Var = []

for i in range(len(P2Data)):

    datasetName = P2Data[i]
    mySize = datasetName.replace("_TEST", "")[1:]
    filteredResultsSummary = resultsSummary[resultsSummary['Data Size'] == mySize].reset_index(drop=True)
    
    print(f"START DATASET: {datasetName}")
    
    for j in range(len(filteredResultsSummary)):
            
        modelName = filteredResultsSummary["Model"].iloc[j]
        dataInstance = filteredResultsSummary["Data Instance"].iloc[j]
        expSol = ast.literal_eval(filteredResultsSummary["Solution"].iloc[j])
        expTime = filteredResultsSummary["Time"].iloc[j]
        expPerCorrect = filteredResultsSummary["Percent Correct"].iloc[j]
        expTotalFeats = filteredResultsSummary["Total Feats"].iloc[j]
        expReplications = filteredResultsSummary["Replications"].iloc[j]
        
        print(f"START MODEL {modelName}-{dataInstance}")
        
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
        
        #####################
        # COMPILE ALL RESULTS
        tempResults = [modelName, dataInstance, mySize, expSol, tempYbar, tempVar, tempPercentLoss, expTime, expReplications, expPerCorrect, expTotalFeats]
        tempColNames = ['Model', 'Data Instance', 'Data Size', 'Solution', 'Loss', 'Variance', 'Percent Loss', "Time", "Replications", "Percent Correct", "Total Feats"]
        tempFinalResults = pd.DataFrame(data = [tempResults], columns = tempColNames)
         
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
            finalP2RESULTS = pd.DataFrame(data = None, columns = tempColNames)
                        
        # Add/Update Results DF
        if len(finalP2RESULTS) == 0:
            # Add row to the end of the dataframe
            finalP2RESULTS = pd.concat([finalP2RESULTS,tempFinalResults], axis = 0, ignore_index=True)
            
        elif ((finalP2RESULTS['Model'] == tempFinalResults['Model'].iloc[0]) & (finalP2RESULTS["Data Instance"] == tempFinalResults['Data Instance'].iloc[0])).any():
                           
            # Replace the existing row
            finalP2RESULTS.loc[(finalP2RESULTS['Model'] == tempFinalResults['Model'].iloc[0]) & (finalP2RESULTS['Data Instance'] == tempFinalResults['Data Instance'].iloc[0])] = tempFinalResults
            
        else:
            finalP2RESULTS = pd.concat([finalP2RESULTS,tempFinalResults], axis = 0, ignore_index=True)
            
        # Export Results
        finalP2RESULTS.to_excel(finalP2RESULTS_Path, index = False)
          

print("Done!")





