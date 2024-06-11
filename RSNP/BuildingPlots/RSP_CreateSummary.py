# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:38:41 2024

@author: ephouser
"""

computer = 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import sys
if computer == 1:
    sys.path.append('G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\')

from RSP_Summary_PrepareData import *

###############################################################################
###############################################################################
##### IMPORT RESULTS

######################
# IMPORT MODEL RESULTS
if computer == 0:
    finalP2RESULTS_Path = f"/home/ephouser/RSP/ResultSummary/Phase2Results.xlsx"
else:
    finalP2RESULTS_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\Phase2Results.xlsx"
    finalP2RESULTS_Path = f"C:\\Users\\ephouser\\Downloads\\Phase2Results.xlsx"

# Import Data
finalP2RESULTS = pd.read_excel(finalP2RESULTS_Path, index_col=None)

###############################################################################
###############################################################################
##### CREATE RESULTS SUMMARY
finalResults = prepareData(finalP2RESULTS)
    
###############
# SEPARATE DATA
finalResults_Loss = finalResults[['Model', 'Feat Size', 'Loss Avg', 'Loss Var']] 
finalResults_PercentLoss = finalResults[['Model', 'Feat Size', 'Percent Loss Avg', 'Percent Loss 10%PI', 'Percent Loss 90%PI']]
finalResults_Time = finalResults[['Model', 'Feat Size', 'Time Avg', 'Time Var']]
finalResults_Replications = finalResults[['Model', 'Feat Size', 'Replications Avg', 'Replications Var']]
finalResults_PercentCorrect = finalResults[['Model', 'Feat Size', 'Percent Correct Avg', 'Percent Correct Var']]
finalResults_TotalFeats = finalResults[['Model', 'Feat Size', 'Total Feats Avg', 'Total Feats 10%PI', 'Total Feats 90%PI']]
       
###############################################################################
###############################################################################
##### GENERATE PLOTS
#####

# Plot Dimensions
width = 6
height = 5

# Color Map
modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 "RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'deeppink',
                 "RS2P-Hybrid-Min-SIZ": 'lightgreen', "RS2P-Shrinking-Min-SIZ": 'lime',
                 "RS3P-Hybrid-Min-SIZ": 'lightblue', "RS3P-Shrinking-Min-SIZ": 'cyan', 
                 "RS20P-Hybrid-Min-SIZ": 'plum', "RS20P-Shrinking-Min-SIZ": 'magenta', 
                 }

# MODEL ORDER
uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', 'ORS20-Shrinking',
                 "RS1P-Hybrid-Min-SIZ", "RS1P-Shrinking-Min-SIZ",
                 "RS2P-Hybrid-Min-SIZ", "RS2P-Shrinking-Min-SIZ",
                 "RS3P-Hybrid-Min-SIZ", "RS3P-Shrinking-Min-SIZ", 
                 "RS20P-Hybrid-Min-SIZ", "RS20P-Shrinking-Min-SIZ"
                 ]

###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
    
'''
##################################################################
##################################################################
####################### 20 TOTAL ITERATIONS ######################
##################################################################
##################################################################
'''

# MODEL ORDER
uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', 'ORS20-Shrinking', 
                "RS20P-Hybrid-Min-SIZ", "RS20P-Shrinking-Min-SIZ"]

# Color Map
modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 "RS20P-Hybrid-Min-SIZ": 'red', "RS20P-Shrinking-Min-SIZ": 'blue'
                 }

#############################
#############################
##     BUILD LOSS PLOTS    ##
##           20 Itr        ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Loss (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


#############################
#############################
## BUILD PERCENT LOSS PLOT ##
##           20 ITR        ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Loss (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


#############################
#############################
##     BUILD TIME PLOTS    ##
##           20ITR         ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Time (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


#############################
#############################
## BUILD REPLICATIONS PLOT ##
##           20 ITR        ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Replications (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('Replciations')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))
    

################################
################################
## BUILD PERCENT CORRECT PLOT ##
##             20 ITR         ##
################################
################################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Correct (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Correct')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


################################
################################
## BUILD PERCENT CORRECT PLOT ##
##             20 ITR         ##
################################
################################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Features (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('# Features')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))   
    plt.xticks(range(15, 151, 15), range(15, 151, 15))
   
    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
 
'''
#################################################################
#################################################################
################ COMPARING TOTAL ITERATION NUMBER ###############
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ"]

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue'
                 }

#############################
#############################
##     BUILD LOSS PLOTS    ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Loss")
    plt.xlabel('Data Size')
    plt.ylabel('Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


#############################
#############################
## BUILD PERCENT LOSS PLOT ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Loss")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


#############################
#############################
##     BUILD TIME PLOTS    ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Time")
    plt.xlabel('Data Size')
    plt.ylabel('Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


#############################
#############################
## BUILD REPLICATIONS PLOT ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Replications")
    plt.xlabel('Data Size')
    plt.ylabel('Replciations')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))
    

################################
################################
## BUILD PERCENT CORRECT PLOT ##
################################
################################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Correct")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Correct')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


################################
################################
## BUILD PERCENT CORRECT PLOT ##
################################
################################

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Features")
    plt.xlabel('Data Size')
    plt.ylabel('# Features')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################


'''
#################################################################
#################################################################
###################### COMPARING ALL MODELS #####################
#################################################################
#################################################################
'''

#############################
#############################
##     BUILD LOSS PLOTS    ##
#############################
#############################

# Filter Data
uniqueModels = finalResults_Loss[["Model"]].drop_duplicates()

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid', 'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Loss (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


#############################
#############################
## BUILD PERCENT LOSS PLOT ##
#############################
#############################

# Filter Data
uniqueModels = finalResults_PercentLoss[["Model"]].drop_duplicates()

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid', 'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Loss (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


#############################
#############################
##     BUILD TIME PLOTS    ##
#############################
#############################

# Filter Data
uniqueModels = finalResults_Time[["Model"]].drop_duplicates()

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid', 'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Time (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


#############################
#############################
## BUILD REPLICATIONS PLOT ##
#############################
#############################

# Filter Data
uniqueModels = finalResults_Replications[["Model"]].drop_duplicates()

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid', 'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }
# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Replications (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('Replciations')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    

################################
################################
## BUILD PERCENT CORRECT PLOT ##
################################
################################

# Filter Data
uniqueModels = finalResults_PercentCorrect[["Model"]].drop_duplicates()

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid', 'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Correct (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Correct')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 


################################
################################
## BUILD PERCENT CORRECT PLOT ##
################################
################################

# Filter Data
uniqueModels = finalResults_TotalFeats[["Model"]].drop_duplicates()

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid', 'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# Initialize the plot
plt.figure(figsize=(width, height))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker='o', label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Features (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('# Features')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
 

# #############################
# #############################
# ##     BUILD LOSS PLOTS    ##
# ##           1 Itr         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Loss[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS1-Hybrid", "RS1P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking',"RS1-Shrinking",  "RS1P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1-Hybrid": 'lightcoral', "RS1-Shrinking": 'lightblue', "RS1P-Hybrid-Min-SIZ": 'red', "RS1P-Shrinking-Min-SIZ": 'blue'}

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Loss (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    
# #############################
# #############################
# ##     BUILD LOSS PLOTS    ##
# ##           2 Itr         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Loss[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS2-Hybrid", "RS2P-Hybrid-Min-SIZ", 
#                 'ORS20-Shrinking', "RS2-Shrinking", "RS2P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS2-Hybrid": 'lightcoral', "RS2-Shrinking": 'lightblue', "RS2P-Hybrid-Min-SIZ": 'red', "RS2P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Loss (2Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
     

# #############################
# #############################
# ##     BUILD LOSS PLOTS    ##
# ##           3 Itr         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Loss[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS3-Hybrid", "RS3P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking', "RS3-Shrinking",  "RS3P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS3-Hybrid": 'lightcoral', "RS3-Shrinking": 'lightblue', "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Loss (3Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

 
    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
    
    
# #############################
# #############################
# ## BUILD PERCENT LOSS PLOT ##
# ##           1 ITR         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_PercentLoss[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS1-Hybrid", "RS1P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking',"RS1-Shrinking",  "RS1P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1-Hybrid": 'lightcoral', "RS1-Shrinking": 'lightblue', "RS1P-Hybrid-Min-SIZ": 'red', "RS1P-Shrinking-Min-SIZ": 'blue'}

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i]  
    
#     # Get Model Data
#     filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Percent Loss (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Percent Loss')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


# #############################
# #############################
# ## BUILD PERCENT LOSS PLOT ##
# ##           2 ITR         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_PercentLoss[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS2-Hybrid", "RS2P-Hybrid-Min-SIZ", 
#                 'ORS20-Shrinking', "RS2-Shrinking", "RS2P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS2-Hybrid": 'lightcoral', "RS2-Shrinking": 'lightblue', "RS2P-Hybrid-Min-SIZ": 'red', "RS2P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Percent Loss (2Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Percent Loss')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    

# #############################
# #############################
# ## BUILD PERCENT LOSS PLOT ##
# ##           3 ITR         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_PercentLoss[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS3-Hybrid", "RS3P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking', "RS3-Shrinking",  "RS3P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS3-Hybrid": 'lightcoral', "RS3-Shrinking": 'lightblue', "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Percent Loss (3Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Percent Loss')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
 
    
# #############################
# #############################
# ##     BUILD TIME PLOTS    ##
# ##           1ITR          ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Time[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS1-Hybrid", "RS1P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking',"RS1-Shrinking",  "RS1P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1-Hybrid": 'lightcoral', "RS1-Shrinking": 'lightblue', "RS1P-Hybrid-Min-SIZ": 'red', "RS1P-Shrinking-Min-SIZ": 'blue'}

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Time (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Time')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


# #############################
# #############################
# ##     BUILD TIME PLOTS    ##
# ##           2ITR          ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Time[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS2-Hybrid", "RS2P-Hybrid-Min-SIZ", 
#                 'ORS20-Shrinking', "RS2-Shrinking", "RS2P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS2-Hybrid": 'lightcoral', "RS2-Shrinking": 'lightblue', "RS2P-Hybrid-Min-SIZ": 'red', "RS2P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Time (2Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Time')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    

# #############################
# #############################
# ##     BUILD TIME PLOTS    ##
# ##           3ITR          ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Time[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS3-Hybrid", "RS3P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking', "RS3-Shrinking",  "RS3P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS3-Hybrid": 'lightcoral', "RS3-Shrinking": 'lightblue', "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Time (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Time')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
        
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
 

# #############################
# #############################
# ## BUILD REPLICATIONS PLOT ##
# ##           1 ITR         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Replications[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS1-Hybrid", "RS1P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking',"RS1-Shrinking",  "RS1P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1-Hybrid": 'lightcoral', "RS1-Shrinking": 'lightblue', "RS1P-Hybrid-Min-SIZ": 'red', "RS1P-Shrinking-Min-SIZ": 'blue'}

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average # of Replications (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Replciations')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    

# #############################
# #############################
# ## BUILD REPLICATIONS PLOT ##
# ##           2 ITR         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Replications[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS2-Hybrid", "RS2P-Hybrid-Min-SIZ", 
#                 'ORS20-Shrinking', "RS2-Shrinking", "RS2P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS2-Hybrid": 'lightcoral', "RS2-Shrinking": 'lightblue', "RS2P-Hybrid-Min-SIZ": 'red', "RS2P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average # of Replications (2Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Replciations')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    
# #############################
# #############################
# ## BUILD REPLICATIONS PLOT ##
# ##           3 ITR         ##
# #############################
# #############################

# # Filter Data
# uniqueModels = finalResults_Replications[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS3-Hybrid", "RS3P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking', "RS3-Shrinking",  "RS3P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS3-Hybrid": 'lightcoral', "RS3-Shrinking": 'lightblue', "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average # of Replications (3Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Replciations')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
   
    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
 

# ################################
# ################################
# ## BUILD PERCENT CORRECT PLOT ##
# ##             1 ITR          ##
# ################################
# ################################

# # Filter Data
# uniqueModels = finalResults_PercentCorrect[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS1-Hybrid", "RS1P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking',"RS1-Shrinking",  "RS1P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1-Hybrid": 'lightcoral', "RS1-Shrinking": 'lightblue', "RS1P-Hybrid-Min-SIZ": 'red', "RS1P-Shrinking-Min-SIZ": 'blue'}

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Percent Correct (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Percent Correct')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))    
   

# ################################
# ################################
# ## BUILD PERCENT CORRECT PLOT ##
# ##             2 ITR          ##
# ################################
# ################################

# # Filter Data
# uniqueModels = finalResults_PercentCorrect[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS2-Hybrid", "RS2P-Hybrid-Min-SIZ", 
#                 'ORS20-Shrinking', "RS2-Shrinking", "RS2P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS2-Hybrid": 'lightcoral', "RS2-Shrinking": 'lightblue', "RS2P-Hybrid-Min-SIZ": 'red', "RS2P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Percent Correct (2Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Percent Correct')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    
    
# ################################
# ################################
# ## BUILD PERCENT CORRECT PLOT ##
# ##             3 ITR          ##
# ################################
# ################################

# # Filter Data
# uniqueModels = finalResults_PercentCorrect[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS3-Hybrid", "RS3P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking', "RS3-Shrinking",  "RS3P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS3-Hybrid": 'lightcoral', "RS3-Shrinking": 'lightblue', "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average Percent Correct (3Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('Percent Correct')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    
       
    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
 
    
# ################################
# ################################
# ## BUILD PERCENT CORRECT PLOT ##
# ##             1 ITR          ##
# ################################
# ################################

# # Filter Data
# uniqueModels = finalResults_TotalFeats[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS1-Hybrid", "RS1P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking',"RS1-Shrinking",  "RS1P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1-Hybrid": 'lightcoral', "RS1-Shrinking": 'lightblue', "RS1P-Hybrid-Min-SIZ": 'red', "RS1P-Shrinking-Min-SIZ": 'blue'}

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average # of Features (1Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('# Features')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))    


# ################################
# ################################
# ## BUILD PERCENT CORRECT PLOT ##
# ##             2 ITR          ##
# ################################
# ################################

# # Filter Data
# uniqueModels = finalResults_TotalFeats[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS2-Hybrid", "RS2P-Hybrid-Min-SIZ", 
#                 'ORS20-Shrinking', "RS2-Shrinking", "RS2P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS2-Hybrid": 'lightcoral', "RS2-Shrinking": 'lightblue', "RS2P-Hybrid-Min-SIZ": 'red', "RS2P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average # of Features (2Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('# Features')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))    

    
# ################################
# ################################
# ## BUILD PERCENT CORRECT PLOT ##
# ##             3 ITR          ##
# ################################
# ################################

# # Filter Data
# uniqueModels = finalResults_TotalFeats[["Model"]].drop_duplicates()

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', "RS3-Hybrid", "RS3P-Hybrid-Min-SIZ",
#                 'ORS20-Shrinking', "RS3-Shrinking",  "RS3P-Shrinking-Min-SIZ"]

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS3-Hybrid": 'lightcoral', "RS3-Shrinking": 'lightblue', "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue'
#                  }

# # Initialize the plot
# plt.figure(figsize=(width, height))
# for i in range(len(uniqueModels)):
       
#     # Get Model Name
#     modelName = uniqueModels[i] 
    
#     # Get Model Data
#     filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
    
#     # Get Color Based on Model
#     color = modelColorMap.get(modelName, 'black')
    
#     # Plot a new line for each iteration
#     plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker='o', label=f'{modelName}', color=color)
   
#     # Add labels and legend
#     plt.title("Average # of Features (3Itr)")
#     plt.xlabel('Data Size')
#     plt.ylabel('# Features')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))    
    
   
    
    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
     
###############################################################################

# # Filter Data
# tempDF = finalResults[['Model', 'Data Size', 'Total Feats 10%PI', 'Total Feats 90%PI']]
# tempDF = tempDF.sort_values(by = ["Model"], ascending= False)

# # Defining the intervals for each category
# intervals_small = tempDF[tempDF['Data Size'] == 'S']
# intervals_medium = tempDF[tempDF['Data Size'] == 'M']
# intervals_large = tempDF[tempDF['Data Size'] == 'L']


# tempPCDF = finalResults[['Model', 'Data Size','Percent Correct Avg']]
# tempPCDF = tempPCDF.sort_values(by = ["Model"], ascending= False)

# labels = { 
#     'S': tempPCDF[tempPCDF['Data Size'] == 'S'],
#     'M': tempPCDF[tempPCDF['Data Size'] == 'M'],
#     'L': tempPCDF[tempPCDF['Data Size'] == 'L']
# }

# font_sizes = [24]*3  # Define font sizes for the titles

# # Creating subplots with adjusted spacing
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26, 9))
# plt.subplots_adjust(wspace=0.1, bottom=0.15)  # Adjust the width and height space between subplots

# # Plotting the lines for each category
# for i, (intervals, size) in enumerate(zip([intervals_small, intervals_medium, intervals_large], ['S', 'M', 'L'])):
#     axs[i].set_title(f'{size} Dataset', fontsize=font_sizes[i], fontweight='bold')  # Set subplot titles with different font sizes
     
#     lstModels = []
#     lstIntervals = []
#     for j in range(len(intervals)):
#         lstModels += [intervals['Model'].iloc[j]]
#         lstIntervals += [list(intervals[['Total Feats 10%PI', 'Total Feats 90%PI']].iloc[j])]
    
#     # Find the lowest and highest values of the intervals
#     lowest_value = min([min(interval) for interval in lstIntervals])
#     highest_value = max([max(interval) for interval in lstIntervals])

#     for spine in axs[i].spines.values():  # Set the color of the plot outline box to black
#         spine.set_edgecolor('black')
    
#     for j in range(len(lstModels)):
        
#         # Model name
#         modName = lstModels[j]
        
#         # Get Color Based on Model
#         color = modelColorMap.get(modName, 'black')
        
#         axs[i].plot(lstIntervals[j], [j, j], marker='|', linestyle='-', linewidth=4, markersize=18, label=lstModels[j], color = color)
#         label_position = min(max(j, 0.1), len(lstIntervals) - 0.5)  # Ensure label stays within plot bounds
       
#         axs[i].set_yticks(range(len(lstIntervals)))
#         axs[i].set_yticklabels([])  # Remove y-axis labels for all but the left subplot
#         if i == 0:
#             axs[i].set_yticklabels(lstModels, fontsize=19, fontweight='bold')  # Change y-axis label font size for the left subplot

#         axs[i].tick_params(axis='y', labelsize=20)  # Change y-axis tick label font size

#         axs[i].set_xticks([math.floor(lowest_value), round((lowest_value + highest_value) / 2), math.ceil(highest_value)])  # Set x-axis tick markers
#         axs[i].set_xlabel('')  # Set common x-axis label for all subplots
#         axs[i].tick_params(axis='x', labelsize=20)  # Change y-axis tick label font size
        
#         label_position = min(max(j, 0.1), len(lstIntervals) - 0.5)  # Ensure label stays within plot bounds
#         axs[i].annotate(f"({labels[size]['Percent Correct Avg'].iloc[j]*100}%)", ((lstIntervals[j][0] + lstIntervals[j][1]) / 2, label_position), textcoords="offset points", xytext=(0, -26), ha='center', fontsize=17)

# # Adding a title for the overall plot with increased fontsize
# fig.text(0.5, 0.02, 'Range of Selected Features\n(Average % of Correct Selection)', ha='center', va='center', fontsize=25, fontweight='bold')

# plt.show()





















