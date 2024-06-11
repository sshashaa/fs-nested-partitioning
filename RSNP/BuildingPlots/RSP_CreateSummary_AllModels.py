# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:38:41 2024

@author: ephouser
"""

computer = 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import math
import seaborn as sns
import ast

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
    
    dataSmall_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\DS1.csv"
    dataMedium_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\DM1.csv"
    dataLarge_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\Data\\DL1.csv"

# Import Data
finalP2RESULTS = pd.read_excel(finalP2RESULTS_Path, index_col=None)
dataSmall = pd.read_csv(dataSmall_Path, index_col=None)
dataMedium = pd.read_csv(dataMedium_Path, index_col=None)
dataLarge = pd.read_csv(dataLarge_Path, index_col=None)

dataSmall = dataSmall.drop(columns = ["y"])
dataMedium = dataMedium.drop(columns = ["y"])
dataLarge = dataLarge.drop(columns = ["y"])

###############################################################################
###############################################################################
##### CREATE RESULTS SUMMARY
finalP2RESULTS = finalP2RESULTS.dropna().reset_index(drop=True)
finalResults = prepareData(finalP2RESULTS)

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

###############
# SEPARATE DATA
finalResults_Loss = finalResults[['Model', 'Feat Size', 'Loss Avg', 'Loss Var', 'Loss 95% HW', 'Loss 10%Q', 'Loss 50%Q', 'Loss 90%Q']] 

finalResults_PercentLoss = finalResults[['Model', 'Feat Size', 'Percent Loss Avg', 'Percent Loss Var', 'Percent Loss 95% HW', 'Percent Loss 10%Q', 'Percent Loss 50%Q', 'Percent Loss 90%Q']]

finalResults_Time = finalResults[['Model', 'Feat Size', 'Time Avg', 'Time Var', 'Time 95% HW', 'Time 10%Q', 'Time 50%Q', 'Time 90%Q']]

finalResults_Replications = finalResults[['Model', 'Feat Size', 'Replications Avg', 'Replications Var', 'Replications 95% HW', 'Replications 10%Q', 'Replications 50%Q', 'Replications 90%Q']]

finalResults_PercentCorrect = finalResults[['Model', 'Feat Size', 'Percent Correct Avg', 'Percent Correct Var', 'Percent Correct 95% HW', 'Percent Correct 10%Q', 'Percent Correct 50%Q', 'Percent Correct 90%Q']]

finalResults_TotalFeats = finalResults[['Model', 'Feat Size', 'Total Feats Avg', 'Total Feats Var', 'Total Feats 95% HW', 'Total Feats 10%Q', 'Total Feats 50%Q', 'Total Feats 90%Q']]
     

    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################

'''
#################################################################
#################################################################
##################### COMPARING 20ITR MODELS ####################
#####################    FEATURE SELECTION   ####################
#################################################################
#################################################################
'''


# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ",     "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ",     "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ",  "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ",  "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid',            'ORS20-Shrinking',
                'NP-Min-SIZ']
        
for dataSize in finalResults['Data Size'].unique():
    
    # Filter Dataset for Size
    filteredDF = finalP2RESULTS[finalP2RESULTS["Data Size"] == dataSize]
    
    # Get Features
    if dataSize == "S":
        listFeats = list(dataSmall.columns)
    elif dataSize == "M":
        listFeats = list(dataMedium.columns)
    elif dataSize == "L":
        listFeats = list(dataLarge.columns)
        
    
    # Create DataFrame
    featHeatMapValsDF = pd.DataFrame(data = None, columns = listFeats)
        
    # Collect Data
    for myModel in uniqueModels:
        
        tempSols = filteredDF["Solution"][filteredDF["Model"] == myModel]
        
        # Count Frequency of Selection for each Variable
        varFreq = []
        for var in listFeats:
            varCount = 0
            for sol in tempSols:
                if var in ast.literal_eval(sol):
                    varCount += 1
                    
            varFreq.append(varCount)
                    
        
        tempDF = pd.DataFrame(data = [varFreq], columns = featHeatMapValsDF.columns)
        
        featHeatMapValsDF = pd.concat([featHeatMapValsDF, tempDF], axis = 0, ignore_index = True)
        featHeatMapValsDF = featHeatMapValsDF.apply(pd.to_numeric, errors='coerce')
        featHeatMapRows = uniqueModels
    
    
    # Create Label Categories
    categories = {'Inf': [val for val in listFeats if 'x' in val], 
                  'Corr': [val for val in listFeats if 'corr' in val], 
                  'Noise': [val for val in listFeats if 'noise' in val]}
        
    # Set the figure size
    plt.figure(figsize=(15, 6))
    
    # Define the relative widths of the subplots
    width_ratios = [2, 1, 7]
    
    # Create a gridspec to specify the layout and relative sizes of subplots
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios)
    
    # Create subplots for each category
    for i, (category, columns) in enumerate(categories.items()):
        ax = plt.subplot(gs[i])
        
        if i == 0:        
            sns.heatmap(featHeatMapValsDF[columns], cmap="RdYlGn", ax=ax, cbar = False)
            ax.set_yticklabels(featHeatMapRows, rotation = 0, fontsize = 15)
            ax.set_xticks([])
            ax.set_xlabel(f"{category}", fontsize = 15)
            
        elif i == 1:
            sns.heatmap(featHeatMapValsDF[columns], cmap="RdYlGn", ax=ax, cbar = False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{category}", fontsize = 15)
            
        elif i == 2:
            sns.heatmap(featHeatMapValsDF[columns], cmap="RdYlGn", ax=ax, cbar = True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{category}", fontsize = 15)
    

    # Plot Title
    plt.suptitle(f'Heatmap of Selected Features \n ({dataSize} Data)', fontsize = 20)
    
    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################


'''
#################################################################
#################################################################
###################### COMPARING ALL MODELS #####################
######################      VIA MEDIAN      #####################
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ",     "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ",     "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ",  "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ",  "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid',            'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato',    "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red',       "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred',  "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta',          'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# MARKER MAP
modelMarkerMap = {'NP-Min-SIZ': 'D', 'ORS20-Hybrid': 'x', 'ORS20-Shrinking': 'P',
                 "RS1P-Hybrid-Min-SIZ": 'h',  "RS1P-Shrinking-Min-SIZ": 'v',
                 "RS2P-Hybrid-Min-SIZ": 'p', "RS2P-Shrinking-Min-SIZ": '>',
                 "RS3P-Hybrid-Min-SIZ": 's',  "RS3P-Shrinking-Min-SIZ": '<', 
                 "RS20P-Hybrid-Min-SIZ": 'o',      "RS20P-Shrinking-Min-SIZ": '*', 
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss 50%Q'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median Loss (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss 50%Q'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median Percent Loss (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time 50%Q'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median Time (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications 50%Q'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median # of Replications (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct 50%Q'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median Percent Correct (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats 50%Q'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median # of Features (All Models)")
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
######################      VIA AVERAGE     #####################
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ",     "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ",     "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ",  "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ",  "RS20P-Shrinking-Min-SIZ",
                'ORS20-Hybrid',            'ORS20-Shrinking',
                'NP-Min-SIZ']

# Color Map
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato',    "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red',       "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred',  "RS20P-Shrinking-Min-SIZ": 'darkblue',
                 'ORS20-Hybrid': 'magenta',          'ORS20-Shrinking': 'lime',
                 'NP-Min-SIZ': 'black'
                 }

# MARKER MAP
modelMarkerMap = {'NP-Min-SIZ': 'D', 'ORS20-Hybrid': 'x', 'ORS20-Shrinking': 'P',
                 "RS1P-Hybrid-Min-SIZ": 'h',  "RS1P-Shrinking-Min-SIZ": 'v',
                 "RS2P-Hybrid-Min-SIZ": 'p', "RS2P-Shrinking-Min-SIZ": '>',
                 "RS3P-Hybrid-Min-SIZ": 's',  "RS3P-Shrinking-Min-SIZ": '<', 
                 "RS20P-Hybrid-Min-SIZ": 'o',      "RS20P-Shrinking-Min-SIZ": '*', 
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Loss (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Loss (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Time (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Replications (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Correct (All Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average # of Features (All Models)")
    plt.xlabel('Data Size')
    plt.ylabel('# Features')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))

