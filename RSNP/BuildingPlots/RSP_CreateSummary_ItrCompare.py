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

myModels = ['RS1P-Hybrid-Min-SIZ', 'RS1P-Shrinking-Min-SIZ', 
            'RS2P-Hybrid-Min-SIZ', 'RS2P-Shrinking-Min-SIZ', 
            'RS3P-Hybrid-Min-SIZ', 'RS3P-Shrinking-Min-SIZ', 
            'RS20P-Hybrid-Min-SIZ', 'RS20P-Shrinking-Min-SIZ', 
            'RS20-Hybrid_n1', 'RS20-Hybrid_n2', 'RS20-Hybrid_n3', 'RS20-Hybrid_n4', 
            'RS20P-Hybrid_n1-Min-SIZ','RS20P-Hybrid_n2-Min-SIZ', 'RS20P-Hybrid_n3-Min-SIZ', 'RS20P-Hybrid_n4-Min-SIZ']

finalResults_PercentCorrect["Model"].unique()




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
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ"]
        
hybridModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ"]
shrinkModels = ["RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ"]

modelLabels = {"RS1P-Hybrid-Min-SIZ": 'Hybrid (1 Itr)',  "RS1P-Shrinking-Min-SIZ": 'Shrink (1 Itr)',
                "RS2P-Hybrid-Min-SIZ": 'Hybrid (2 Itr)', "RS2P-Shrinking-Min-SIZ": 'Shrink (2 Itr)',
                "RS3P-Hybrid-Min-SIZ": 'Hybrid (3 Itr)',  "RS3P-Shrinking-Min-SIZ": 'Shrink (3 Itr)', 
                "RS20P-Hybrid-Min-SIZ": 'Hybrid (20 Itr)',      "RS20P-Shrinking-Min-SIZ": 'Shrink (20 Itr)', 
                 }

modelType = ["Hybrid", "Shrinking"]

for dataSize in finalResults['Data Size'].unique():
    
    # Filter Dataset for Size
    filteredDF = finalP2RESULTS[finalP2RESULTS["Data Size"] == dataSize]
    
    # Get Features
    if dataSize == "S":
        listFeats = list(dataSmall.columns)
        mySize = "Small"
        width_ratios = [3, 2, 10]

    elif dataSize == "M":
        listFeats = list(dataMedium.columns)
        mySize = "Medium"
        width_ratios = [6, 4, 50]
        
    elif dataSize == "L":
        listFeats = list(dataLarge.columns)
        mySize = "Large"
        width_ratios = [10, 5, 135]
        
    
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
    
    # Create Label Categories
    categories = {'True': [val for val in listFeats if 'x' in val], 
                  'Corr': [val for val in listFeats if 'corr' in val], 
                  'Noise': [val for val in listFeats if 'noise' in val]}
        
    # Set the figure size
    plt.figure(figsize=(18, 6))
    
    # Create a gridspec to specify the layout and relative sizes of subplots
    gs = gridspec.GridSpec(2, 3, width_ratios=width_ratios)
    
    hybridSort = [3, 2, 1, 0]
    shrinkSort = [7, 6, 5, 4]
    
    # Create subplots for each category
    for searchIDX in range(2):
        
        searchType = modelType[searchIDX]
        rows = [idx for idx in range(len(uniqueModels)) if searchType in uniqueModels[idx]]
        featHeatMapRows = [uniqueModels[idx] for idx in rows]
        tempLabels = [modelLabels[name] for name in featHeatMapRows]
        
        sortGroup = hybridSort if searchIDX == 0 else shrinkSort
        
        for i, (category, columns) in enumerate(categories.items()):
            ax = plt.subplot(gs[searchIDX, i])
            
            # Sort Data
            featHeatMapValsDFSorted = featHeatMapValsDF[columns].T
            featHeatMapValsDFSorted = featHeatMapValsDFSorted.sort_values(by = sortGroup, ascending=False)
            featHeatMapValsDFSorted = featHeatMapValsDFSorted.T
            
            if i == 0:        
                sns.heatmap(featHeatMapValsDFSorted.loc[rows], cmap="Blues", ax=ax, cbar = False)
                ax.set_yticklabels(tempLabels, rotation = 0, fontsize = 24)
                ax.set_xticks([])                
                if searchIDX == 1:
                    ax.set_xlabel(f"{category}", fontsize = 24)
                
            elif i == 1:
                sns.heatmap(featHeatMapValsDFSorted.loc[rows], cmap="Blues", ax=ax, cbar = False)
                ax.set_xticks([])
                ax.set_yticks([])
                if searchIDX == 1:
                    ax.set_xlabel(f"{category}", fontsize = 24)
                
            elif i == 2:
                
                sns.heatmap(featHeatMapValsDFSorted.loc[rows], cmap="Blues", ax=ax, cbar = False)
                ax.set_xticks([])
                ax.set_yticks([])
                if searchIDX == 1:
                    ax.set_xlabel(f"{category}", fontsize = 24)
                
            # Draw borders around subplots
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
    
            # Draw borders around subplots
            spineWidth = 3
            ax.spines['top'].set_linewidth(spineWidth)
            ax.spines['right'].set_linewidth(spineWidth)
            ax.spines['bottom'].set_linewidth(spineWidth)
            ax.spines['left'].set_linewidth(spineWidth)

    # Add a color bar for the entire figure
    cbar_ax = plt.gcf().add_axes([1.02, 0.125, 0.03, 0.7])
    cbar = plt.colorbar(ax.collections[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

    # Plot Title
    plt.suptitle(f'Heatmap of Selected Features in the {mySize} Dataset', fontsize = 24)
    
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
################## COMPARING TOTAL NUM ITERATIONS ###############
##################           VIA MEDIAN           ###############
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ"]

# COLOR MAP
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue'
                 }

# MARKER MAP
modelMarkerMap = {"RS1P-Hybrid-Min-SIZ": 'o', "RS1P-Shrinking-Min-SIZ": 'v',
                  "RS2P-Hybrid-Min-SIZ": '*', "RS2P-Shrinking-Min-SIZ": 'p',
                  "RS3P-Hybrid-Min-SIZ": 's', "RS3P-Shrinking-Min-SIZ": 'o',
                  "RS20P-Hybrid-Min-SIZ": 'd', "RS20P-Shrinking-Min-SIZ": 'o'
                 }

# MARKER MAP
modelMarkerMap = {"RS1P-Hybrid-Min-SIZ": 'h',  "RS1P-Shrinking-Min-SIZ": 'v',
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
    plt.title("Median Loss ")
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
    plt.title("Median Percent Loss ")
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
    plt.title("Median Time ")
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
    plt.title("Median # of Replications ")
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
    plt.title("Median Percent Correct ")
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
    plt.title("Median # of Features ")
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
################## COMPARING TOTAL NUM ITERATIONS ###############
##################              VIA AVG           ###############
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Loss Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Replications Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Correct")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Correct')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


################################
################################
## BUILD TOTAL FEATURES PLOT  ##
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
################## COMPARING TOTAL NUM ITERATIONS ###############
##################           (PI BOUNDS)          ###############
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ["RS1P-Hybrid-Min-SIZ", "RS2P-Hybrid-Min-SIZ", 
                "RS3P-Hybrid-Min-SIZ", "RS20P-Hybrid-Min-SIZ", 
                "RS1P-Shrinking-Min-SIZ", "RS2P-Shrinking-Min-SIZ", 
                "RS3P-Shrinking-Min-SIZ", "RS20P-Shrinking-Min-SIZ"]

# COLOR MAP
modelColorMap = {"RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'lightblue',
                 "RS2P-Hybrid-Min-SIZ": 'tomato', "RS2P-Shrinking-Min-SIZ": 'deepskyblue',
                 "RS3P-Hybrid-Min-SIZ": 'red', "RS3P-Shrinking-Min-SIZ": 'blue',
                 "RS20P-Hybrid-Min-SIZ": 'darkred', "RS20P-Shrinking-Min-SIZ": 'darkblue'
                 }

# MARKER MAP
modelMarkerMap = {"RS1P-Hybrid-Min-SIZ": 'o', "RS1P-Shrinking-Min-SIZ": 'v',
                  "RS2P-Hybrid-Min-SIZ": '*', "RS2P-Shrinking-Min-SIZ": 'p',
                  "RS3P-Hybrid-Min-SIZ": 's', "RS3P-Shrinking-Min-SIZ": 'o',
                  "RS20P-Hybrid-Min-SIZ": 'd', "RS20P-Shrinking-Min-SIZ": 'o'
                 }

# MARKER MAP
modelMarkerMap = {"RS1P-Hybrid-Min-SIZ": 'h',  "RS1P-Shrinking-Min-SIZ": 'v',
                  "RS2P-Hybrid-Min-SIZ": 'p', "RS2P-Shrinking-Min-SIZ": '>',
                  "RS3P-Hybrid-Min-SIZ": 's',  "RS3P-Shrinking-Min-SIZ": '<', 
                  "RS20P-Hybrid-Min-SIZ": 'o',      "RS20P-Shrinking-Min-SIZ": '*', 
                 }

'''
################################################################
################################################################
################################################################
##############           BUILD LOSS PLOT          ##############
################################################################
################################################################
################################################################

############################
############################
##### PI PLOTS BY SIZE #####
############################
############################
'''

# General variables
title_contains_loss = True
performance_metric_dataset = finalResults_Loss
performance_metric_name = "Loss"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"
performance_metric_ylabel = "Hybrid Search Procedure"
other_metric_ylabel = "Shrinking Search Procedure"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Define model groups
hybrid_models = [model for model in uniqueModels if 'Hybrid' in model]
shrinking_models = [model for model in uniqueModels if 'Shrinking' in model]

# Create a figure with two rows and three columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Set title for the middle subplot of each row and increase font size
if title_contains_loss:
    axs[0, 1].set_title(performance_metric_title, fontsize=18)

# Initialize variables to store min and max x-axis values for each column
min_x = [[float('inf') for _ in range(3)] for _ in range(2)]
max_x = [[float('-inf') for _ in range(3)] for _ in range(2)]

# Plot each subplot for hybrid models
for i, dataset_size in enumerate(dataset_sizes):
    for j, modelName in enumerate(uniqueModels):
        
        if modelName in hybrid_models:
            rowIDX = 0
        elif modelName in shrinking_models:
            rowIDX = 1
    
        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == dataset_size)]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the prediction interval (10%-90% quantile)
        interval = axs[rowIDX, i].fill_betweenx([modelName]*len(filteredModelData), 
                                                 filteredModelData[f'{performance_metric_name} 10%Q'], 
                                                 filteredModelData[f'{performance_metric_name} 90%Q'], 
                                                 alpha=1, color=color, label=f'{modelName.split("-")[0]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        # Update min and max x-axis values for each column
        min_x[rowIDX][i] = min(min_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 10%Q'].min())
        max_x[rowIDX][i] = max(max_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 90%Q'].max())

# Set the same x-axis limits for all subplots in each column
for i in range(3):
    min_x_col = min(min_x[0][i], min_x[1][i])
    max_x_col = max(max_x[0][i], max_x[1][i])
    
    buffer = (max_x_col - min_x_col) * 0.1
    
    for rowIDX in range(2):
        axs[rowIDX, i].set_xlim(min_x_col - buffer, max_x_col + buffer)
        
        # Adjust the spacing between y-axis tick marks
        axs[rowIDX, i].set_yticks(axs[rowIDX, i].get_yticks()[::2])  # Adjust the step size as needed
    
    if dataset_sizes[i] == 15:
        mySize = "Small"
    elif dataset_sizes[i] == 60:
        mySize = "Medium"
    elif dataset_sizes[i] == 150:
        mySize = "Large"
    
    # Set x-axis ticks at the bottom for the top row and at the top for the bottom row
    if rowIDX == 0:
        axs[rowIDX, i].xaxis.set_ticks_position('bottom')
    else:
        axs[rowIDX, i].xaxis.set_ticks_position('top')
    
    # Increase x-axis label font size and ensure at least 5 ticks
    axs[1,i].set_xlabel(f"{mySize} Dataset \n ({dataset_sizes[i]} Features)", fontsize=14)
    
    # Set the number of tick marks on the x axis
    axs[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axs[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Add padding between x-axis tick labels and the plot
    axs[1, i].tick_params(axis='x', pad=10, labelsize=12)

# Remove y-axis tick marks for the middle and far right subplots of each row
for i in range(2):
    for j in range(1, 3):
        axs[i, j].set_yticklabels([])
        
for i in range(3):
    axs[0, i].set_xticklabels([])
        
# Add Y Axis Titles and increase font size
axs[0, 0].set_ylabel(performance_metric_ylabel, fontsize=14)
axs[1, 0].set_ylabel(other_metric_ylabel, fontsize=14)

# Add a single legend for all subplots
axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
axs[1, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

for i in range(2):
    # Get handles and labels of legend items
    handles, labels = axs[i, 2].get_legend_handles_labels()
    
    # Reverse the order
    handles.reverse()
    labels.reverse()
    
    # Create a new legend with the reversed order
    axs[i, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.tight_layout()
plt.show()



'''
#############################
#############################
##### PI PLOTS BY MODEL #####
#############################
#############################
'''

# Create a figure with three subplots in a horizontal layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each subplot
axs[1].set_xlabel('Data Size')
axs[0].set_ylabel('Average Loss')
for i, quantile in enumerate(['Loss 10%Q', 'Loss 50%Q', 'Loss 90%Q']):
    
    titleSegments = [' '.join(quantile.split(' ')[0:-1]), quantile.split(' ')[-1]]
    axs[i].set_title(f"{titleSegments[0]} ({titleSegments[1].replace('Q', '')} Quantile)")
    
    for j, modelName in enumerate(uniqueModels):
        # Get Model Data
        filteredModelData = finalResults_Loss[finalResults_Loss['Model'] == modelName]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the data for the current model
        axs[i].plot(filteredModelData['Feat Size'], filteredModelData[quantile], 
                    marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)

# Add a single legend for all subplots
axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()



'''
################################################################
################################################################
################################################################
##############       BUILD PERCENT LOSS PLOT      ##############
################################################################
################################################################
################################################################


############################
############################
##### PI PLOTS BY SIZE #####
############################
############################
'''

# General variables
title_contains_loss = True
performance_metric_dataset = finalResults_PercentLoss
performance_metric_name = "Percent Loss"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"
performance_metric_ylabel = "Hybrid Search Procedure"
other_metric_ylabel = "Shrinking Search Procedure"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Define model groups
hybrid_models = [model for model in uniqueModels if 'Hybrid' in model]
shrinking_models = [model for model in uniqueModels if 'Shrinking' in model]

# Create a figure with two rows and three columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Set title for the middle subplot of each row and increase font size
if title_contains_loss:
    axs[0, 1].set_title(performance_metric_title, fontsize=18)

# Initialize variables to store min and max x-axis values for each column
min_x = [[float('inf') for _ in range(3)] for _ in range(2)]
max_x = [[float('-inf') for _ in range(3)] for _ in range(2)]

# Plot each subplot for hybrid models
for i, dataset_size in enumerate(dataset_sizes):
    for j, modelName in enumerate(uniqueModels):
        
        if modelName in hybrid_models:
            rowIDX = 0
        elif modelName in shrinking_models:
            rowIDX = 1
    
        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == dataset_size)]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the prediction interval (10%-90% quantile)
        interval = axs[rowIDX, i].fill_betweenx([modelName]*len(filteredModelData), 
                                                 filteredModelData[f'{performance_metric_name} 10%Q'], 
                                                 filteredModelData[f'{performance_metric_name} 90%Q'], 
                                                 alpha=1, color=color, label=f'{modelName.split("-")[0]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        # Update min and max x-axis values for each column
        min_x[rowIDX][i] = min(min_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 10%Q'].min())
        max_x[rowIDX][i] = max(max_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 90%Q'].max())

# Set the same x-axis limits for all subplots in each column
for i in range(3):
    min_x_col = min(min_x[0][i], min_x[1][i])
    max_x_col = max(max_x[0][i], max_x[1][i])
    
    buffer = (max_x_col - min_x_col) * 0.1
    
    for rowIDX in range(2):
        axs[rowIDX, i].set_xlim(min_x_col - buffer, max_x_col + buffer)
        
        # Adjust the spacing between y-axis tick marks
        axs[rowIDX, i].set_yticks(axs[rowIDX, i].get_yticks()[::2])  # Adjust the step size as needed
    
    if dataset_sizes[i] == 15:
        mySize = "Small"
    elif dataset_sizes[i] == 60:
        mySize = "Medium"
    elif dataset_sizes[i] == 150:
        mySize = "Large"
    
    # Set x-axis ticks at the bottom for the top row and at the top for the bottom row
    if rowIDX == 0:
        axs[rowIDX, i].xaxis.set_ticks_position('bottom')
    else:
        axs[rowIDX, i].xaxis.set_ticks_position('top')
    
    # Increase x-axis label font size and ensure at least 5 ticks
    axs[1,i].set_xlabel(f"{mySize} Dataset \n ({dataset_sizes[i]} Features)", fontsize=14)
    
    # Set the number of tick marks on the x axis
    axs[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axs[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Add padding between x-axis tick labels and the plot
    axs[1, i].tick_params(axis='x', pad=10, labelsize=12)

# Remove y-axis tick marks for the middle and far right subplots of each row
for i in range(2):
    for j in range(1, 3):
        axs[i, j].set_yticklabels([])
        
for i in range(3):
    axs[0, i].set_xticklabels([])
        
# Add Y Axis Titles and increase font size
axs[0, 0].set_ylabel(performance_metric_ylabel, fontsize=14)
axs[1, 0].set_ylabel(other_metric_ylabel, fontsize=14)

# Add a single legend for all subplots
axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
axs[1, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

for i in range(2):
    # Get handles and labels of legend items
    handles, labels = axs[i, 2].get_legend_handles_labels()
    
    # Reverse the order
    handles.reverse()
    labels.reverse()
    
    # Create a new legend with the reversed order
    axs[i, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.tight_layout()
plt.show()



'''
#############################
#############################
##### PI PLOTS BY MODEL #####
#############################
#############################
'''

# Create a figure with three subplots in a horizontal layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each subplot
axs[1].set_xlabel('Data Size')
axs[0].set_ylabel('Average Percent Loss')
for i, quantile in enumerate(['Percent Loss 10%Q', 'Percent Loss 50%Q', 'Percent Loss 90%Q']):
    
    titleSegments = [' '.join(quantile.split(' ')[0:-1]), quantile.split(' ')[-1]]
    axs[i].set_title(f"{titleSegments[0]} ({titleSegments[1].replace('Q', '')} Quantile)")
    
    for j, modelName in enumerate(uniqueModels):
        # Get Model Data
        filteredModelData = finalResults_PercentLoss[finalResults_PercentLoss['Model'] == modelName]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the data for the current model
        axs[i].plot(filteredModelData['Feat Size'], filteredModelData[quantile], 
                    marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)

# Add a single legend for all subplots
axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


'''
################################################################
################################################################
################################################################
##############           BUILD TIME PLOT          ##############
################################################################
################################################################
################################################################


############################
############################
##### PI PLOTS BY SIZE #####
############################
############################
'''

# General variables
title_contains_loss = True
performance_metric_dataset = finalResults_Time
performance_metric_name = "Time"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"
performance_metric_ylabel = "Hybrid Search Procedure"
other_metric_ylabel = "Shrinking Search Procedure"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Define model groups
hybrid_models = [model for model in uniqueModels if 'Hybrid' in model]
shrinking_models = [model for model in uniqueModels if 'Shrinking' in model]

# Create a figure with two rows and three columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Set title for the middle subplot of each row and increase font size
if title_contains_loss:
    axs[0, 1].set_title(performance_metric_title, fontsize=18)

# Initialize variables to store min and max x-axis values for each column
min_x = [[float('inf') for _ in range(3)] for _ in range(2)]
max_x = [[float('-inf') for _ in range(3)] for _ in range(2)]

# Plot each subplot for hybrid models
for i, dataset_size in enumerate(dataset_sizes):
    for j, modelName in enumerate(uniqueModels):
        
        if modelName in hybrid_models:
            rowIDX = 0
        elif modelName in shrinking_models:
            rowIDX = 1
    
        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == dataset_size)]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the prediction interval (10%-90% quantile)
        interval = axs[rowIDX, i].fill_betweenx([modelName]*len(filteredModelData), 
                                                 filteredModelData[f'{performance_metric_name} 10%Q'], 
                                                 filteredModelData[f'{performance_metric_name} 90%Q'], 
                                                 alpha=1, color=color, label=f'{modelName.split("-")[0]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        # Update min and max x-axis values for each column
        min_x[rowIDX][i] = min(min_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 10%Q'].min())
        max_x[rowIDX][i] = max(max_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 90%Q'].max())

# Set the same x-axis limits for all subplots in each column
for i in range(3):
    min_x_col = min(min_x[0][i], min_x[1][i])
    max_x_col = max(max_x[0][i], max_x[1][i])
    
    buffer = (max_x_col - min_x_col) * 0.1
    
    for rowIDX in range(2):
        axs[rowIDX, i].set_xlim(min_x_col - buffer, max_x_col + buffer)
        
        # Adjust the spacing between y-axis tick marks
        axs[rowIDX, i].set_yticks(axs[rowIDX, i].get_yticks()[::2])  # Adjust the step size as needed
    
    if dataset_sizes[i] == 15:
        mySize = "Small"
    elif dataset_sizes[i] == 60:
        mySize = "Medium"
    elif dataset_sizes[i] == 150:
        mySize = "Large"
    
    # Set x-axis ticks at the bottom for the top row and at the top for the bottom row
    if rowIDX == 0:
        axs[rowIDX, i].xaxis.set_ticks_position('bottom')
    else:
        axs[rowIDX, i].xaxis.set_ticks_position('top')
    
    # Increase x-axis label font size and ensure at least 5 ticks
    axs[1,i].set_xlabel(f"{mySize} Dataset \n ({dataset_sizes[i]} Features)", fontsize=14)
    
    # Set the number of tick marks on the x axis
    axs[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axs[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Add padding between x-axis tick labels and the plot
    axs[1, i].tick_params(axis='x', pad=10, labelsize=12)

# Remove y-axis tick marks for the middle and far right subplots of each row
for i in range(2):
    for j in range(1, 3):
        axs[i, j].set_yticklabels([])
        
for i in range(3):
    axs[0, i].set_xticklabels([])
        
# Add Y Axis Titles and increase font size
axs[0, 0].set_ylabel(performance_metric_ylabel, fontsize=14)
axs[1, 0].set_ylabel(other_metric_ylabel, fontsize=14)

# Add a single legend for all subplots
axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
axs[1, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

for i in range(2):
    # Get handles and labels of legend items
    handles, labels = axs[i, 2].get_legend_handles_labels()
    
    # Reverse the order
    handles.reverse()
    labels.reverse()
    
    # Create a new legend with the reversed order
    axs[i, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.tight_layout()
plt.show()




'''
#############################
#############################
##### PI PLOTS BY MODEL #####
#############################
#############################
'''

# Create a figure with three subplots in a horizontal layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each subplot
axs[1].set_xlabel('Data Size')
axs[0].set_ylabel('Average Time')
for i, quantile in enumerate(['Time 10%Q', 'Time 50%Q', 'Time 90%Q']):
    
    titleSegments = [' '.join(quantile.split(' ')[0:-1]), quantile.split(' ')[-1]]
    axs[i].set_title(f"{titleSegments[0]} ({titleSegments[1].replace('Q', '')} Quantile)")
    
    for j, modelName in enumerate(uniqueModels):
        # Get Model Data
        filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the data for the current model
        axs[i].plot(filteredModelData['Feat Size'], filteredModelData[quantile], 
                    marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)

# Add a single legend for all subplots
axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()



'''
################################################################
################################################################
################################################################
##############       BUILD REPLICATIONS PLOT      ##############
################################################################
################################################################
################################################################


############################
############################
##### PI PLOTS BY SIZE #####
############################
############################
'''

# General variables
title_contains_loss = True
performance_metric_dataset = finalResults_Replications
performance_metric_name = "Replications"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"
performance_metric_ylabel = "Hybrid Search Procedure"
other_metric_ylabel = "Shrinking Search Procedure"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Define model groups
hybrid_models = [model for model in uniqueModels if 'Hybrid' in model]
shrinking_models = [model for model in uniqueModels if 'Shrinking' in model]

# Create a figure with two rows and three columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Set title for the middle subplot of each row and increase font size
if title_contains_loss:
    axs[0, 1].set_title(performance_metric_title, fontsize=18)

# Initialize variables to store min and max x-axis values for each column
min_x = [[float('inf') for _ in range(3)] for _ in range(2)]
max_x = [[float('-inf') for _ in range(3)] for _ in range(2)]

# Plot each subplot for hybrid models
for i, dataset_size in enumerate(dataset_sizes):
    for j, modelName in enumerate(uniqueModels):
        
        if modelName in hybrid_models:
            rowIDX = 0
        elif modelName in shrinking_models:
            rowIDX = 1
    
        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == dataset_size)]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the prediction interval (10%-90% quantile)
        interval = axs[rowIDX, i].fill_betweenx([modelName]*len(filteredModelData), 
                                                 filteredModelData[f'{performance_metric_name} 10%Q'], 
                                                 filteredModelData[f'{performance_metric_name} 90%Q'], 
                                                 alpha=1, color=color, label=f'{modelName.split("-")[0]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        # Update min and max x-axis values for each column
        min_x[rowIDX][i] = min(min_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 10%Q'].min())
        max_x[rowIDX][i] = max(max_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 90%Q'].max())

# Set the same x-axis limits for all subplots in each column
for i in range(3):
    min_x_col = min(min_x[0][i], min_x[1][i])
    max_x_col = max(max_x[0][i], max_x[1][i])
    
    buffer = (max_x_col - min_x_col) * 0.1
    
    for rowIDX in range(2):
        axs[rowIDX, i].set_xlim(min_x_col - buffer, max_x_col + buffer)
        
        # Adjust the spacing between y-axis tick marks
        axs[rowIDX, i].set_yticks(axs[rowIDX, i].get_yticks()[::2])  # Adjust the step size as needed
    
    if dataset_sizes[i] == 15:
        mySize = "Small"
    elif dataset_sizes[i] == 60:
        mySize = "Medium"
    elif dataset_sizes[i] == 150:
        mySize = "Large"
    
    # Set x-axis ticks at the bottom for the top row and at the top for the bottom row
    if rowIDX == 0:
        axs[rowIDX, i].xaxis.set_ticks_position('bottom')
    else:
        axs[rowIDX, i].xaxis.set_ticks_position('top')
    
    # Increase x-axis label font size and ensure at least 5 ticks
    axs[1,i].set_xlabel(f"{mySize} Dataset \n ({dataset_sizes[i]} Features)", fontsize=14)
    
    # Set the number of tick marks on the x axis
    axs[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axs[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Add padding between x-axis tick labels and the plot
    axs[1, i].tick_params(axis='x', pad=10, labelsize=12)

# Remove y-axis tick marks for the middle and far right subplots of each row
for i in range(2):
    for j in range(1, 3):
        axs[i, j].set_yticklabels([])
        
for i in range(3):
    axs[0, i].set_xticklabels([])
        
# Add Y Axis Titles and increase font size
axs[0, 0].set_ylabel(performance_metric_ylabel, fontsize=14)
axs[1, 0].set_ylabel(other_metric_ylabel, fontsize=14)

# Add a single legend for all subplots
axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
axs[1, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

for i in range(2):
    # Get handles and labels of legend items
    handles, labels = axs[i, 2].get_legend_handles_labels()
    
    # Reverse the order
    handles.reverse()
    labels.reverse()
    
    # Create a new legend with the reversed order
    axs[i, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.tight_layout()
plt.show()




'''
#############################
#############################
##### PI PLOTS BY MODEL #####
#############################
#############################
'''

# Create a figure with three subplots in a horizontal layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each subplot
axs[1].set_xlabel('Data Size')
axs[0].set_ylabel('Average Replications')
for i, quantile in enumerate(['Replications 10%Q', 'Replications 50%Q', 'Replications 90%Q']):
    
    titleSegments = [' '.join(quantile.split(' ')[0:-1]), quantile.split(' ')[-1]]
    axs[i].set_title(f"{titleSegments[0]} ({titleSegments[1].replace('Q', '')} Quantile)")
    
    for j, modelName in enumerate(uniqueModels):
        # Get Model Data
        filteredModelData = finalResults_Replications[finalResults_Replications['Model'] == modelName]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the data for the current model
        axs[i].plot(filteredModelData['Feat Size'], filteredModelData[quantile], 
                    marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)

# Add a single legend for all subplots
axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


'''
################################################################
################################################################
################################################################
##############     BUILD PERCENT CORRECT PLOT     ##############
################################################################
################################################################
################################################################


############################
############################
##### PI PLOTS BY SIZE #####
############################
############################
'''

# General variables
title_contains_loss = True
performance_metric_dataset = finalResults_PercentCorrect
performance_metric_name = "Percent Correct"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"
performance_metric_ylabel = "Hybrid Search Procedure"
other_metric_ylabel = "Shrinking Search Procedure"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Define model groups
hybrid_models = [model for model in uniqueModels if 'Hybrid' in model]
shrinking_models = [model for model in uniqueModels if 'Shrinking' in model]

# Create a figure with two rows and three columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Set title for the middle subplot of each row and increase font size
if title_contains_loss:
    axs[0, 1].set_title(performance_metric_title, fontsize=18)

# Initialize variables to store min and max x-axis values for each column
min_x = [[float('inf') for _ in range(3)] for _ in range(2)]
max_x = [[float('-inf') for _ in range(3)] for _ in range(2)]

# Plot each subplot for hybrid models
for i, dataset_size in enumerate(dataset_sizes):
    for j, modelName in enumerate(uniqueModels):
        
        if modelName in hybrid_models:
            rowIDX = 0
        elif modelName in shrinking_models:
            rowIDX = 1
    
        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == dataset_size)]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the prediction interval (10%-90% quantile)
        interval = axs[rowIDX, i].fill_betweenx([modelName]*len(filteredModelData), 
                                                 filteredModelData[f'{performance_metric_name} 10%Q'], 
                                                 filteredModelData[f'{performance_metric_name} 90%Q'], 
                                                 alpha=1, color=color, label=f'{modelName.split("-")[0]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        # Update min and max x-axis values for each column
        min_x[rowIDX][i] = min(min_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 10%Q'].min())
        max_x[rowIDX][i] = max(max_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 90%Q'].max())

# Set the same x-axis limits for all subplots in each column
for i in range(3):
    min_x_col = min(min_x[0][i], min_x[1][i])
    max_x_col = max(max_x[0][i], max_x[1][i])
    
    buffer = (max_x_col - min_x_col) * 0.1
    
    for rowIDX in range(2):
        axs[rowIDX, i].set_xlim(min_x_col - buffer, max_x_col + buffer)
        
        # Adjust the spacing between y-axis tick marks
        axs[rowIDX, i].set_yticks(axs[rowIDX, i].get_yticks()[::2])  # Adjust the step size as needed
    
    if dataset_sizes[i] == 15:
        mySize = "Small"
    elif dataset_sizes[i] == 60:
        mySize = "Medium"
    elif dataset_sizes[i] == 150:
        mySize = "Large"
    
    # Set x-axis ticks at the bottom for the top row and at the top for the bottom row
    if rowIDX == 0:
        axs[rowIDX, i].xaxis.set_ticks_position('bottom')
    else:
        axs[rowIDX, i].xaxis.set_ticks_position('top')
    
    # Increase x-axis label font size and ensure at least 5 ticks
    axs[1,i].set_xlabel(f"{mySize} Dataset \n ({dataset_sizes[i]} Features)", fontsize=14)
    
    # Set the number of tick marks on the x axis
    axs[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axs[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Add padding between x-axis tick labels and the plot
    axs[1, i].tick_params(axis='x', pad=10, labelsize=12)

# Remove y-axis tick marks for the middle and far right subplots of each row
for i in range(2):
    for j in range(1, 3):
        axs[i, j].set_yticklabels([])
        
for i in range(3):
    axs[0, i].set_xticklabels([])
        
# Add Y Axis Titles and increase font size
axs[0, 0].set_ylabel(performance_metric_ylabel, fontsize=14)
axs[1, 0].set_ylabel(other_metric_ylabel, fontsize=14)

# Add a single legend for all subplots
axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
axs[1, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

for i in range(2):
    # Get handles and labels of legend items
    handles, labels = axs[i, 2].get_legend_handles_labels()
    
    # Reverse the order
    handles.reverse()
    labels.reverse()
    
    # Create a new legend with the reversed order
    axs[i, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.tight_layout()
plt.show()


'''
#############################
#############################
##### PI PLOTS BY MODEL #####
#############################
#############################
'''

# Create a figure with three subplots in a horizontal layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each subplot
axs[1].set_xlabel('Data Size')
axs[0].set_ylabel('Average Percent Correct')
for i, quantile in enumerate(['Percent Correct 10%Q', 'Percent Correct 50%Q', 'Percent Correct 90%Q']):
    
    titleSegments = [' '.join(quantile.split(' ')[0:-1]), quantile.split(' ')[-1]]
    axs[i].set_title(f"{titleSegments[0]} ({titleSegments[1].replace('Q', '')} Quantile)")
    
    for j, modelName in enumerate(uniqueModels):
        # Get Model Data
        filteredModelData = finalResults_PercentCorrect[finalResults_PercentCorrect['Model'] == modelName]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the data for the current model
        axs[i].plot(filteredModelData['Feat Size'], filteredModelData[quantile], 
                    marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)

# Add a single legend for all subplots
axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


'''
################################################################
################################################################
################################################################
##############     BUILD TOTAL FEATURES PLOT      ##############
################################################################
################################################################
################################################################


############################
############################
##### PI PLOTS BY SIZE #####
############################
############################
'''

# General variables
title_contains_loss = True
performance_metric_dataset = finalResults_TotalFeats
performance_metric_name = "Total Feats"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"
performance_metric_ylabel = "Hybrid Search Procedure"
other_metric_ylabel = "Shrinking Search Procedure"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Define model groups
hybrid_models = [model for model in uniqueModels if 'Hybrid' in model]
shrinking_models = [model for model in uniqueModels if 'Shrinking' in model]

# Create a figure with two rows and three columns of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')

# Set title for the middle subplot of each row and increase font size
if title_contains_loss:
    axs[0, 1].set_title(performance_metric_title, fontsize=18)

# Initialize variables to store min and max x-axis values for each column
min_x = [[float('inf') for _ in range(3)] for _ in range(2)]
max_x = [[float('-inf') for _ in range(3)] for _ in range(2)]

# Plot each subplot for hybrid models
for i, dataset_size in enumerate(dataset_sizes):
    for j, modelName in enumerate(uniqueModels):
        
        if modelName in hybrid_models:
            rowIDX = 0
        elif modelName in shrinking_models:
            rowIDX = 1
    
        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == dataset_size)]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the prediction interval (10%-90% quantile)
        interval = axs[rowIDX, i].fill_betweenx([modelName]*len(filteredModelData), 
                                                 filteredModelData[f'{performance_metric_name} 10%Q'], 
                                                 filteredModelData[f'{performance_metric_name} 90%Q'], 
                                                 alpha=1, color=color, label=f'{modelName.split("-")[0]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        axs[rowIDX, i].plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        # Update min and max x-axis values for each column
        min_x[rowIDX][i] = min(min_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 10%Q'].min())
        max_x[rowIDX][i] = max(max_x[rowIDX][i], filteredModelData[f'{performance_metric_name} 90%Q'].max())

# Set the same x-axis limits for all subplots in each column
for i in range(3):
    min_x_col = min(min_x[0][i], min_x[1][i])
    max_x_col = max(max_x[0][i], max_x[1][i])
    
    buffer = (max_x_col - min_x_col) * 0.1
    
    for rowIDX in range(2):
        axs[rowIDX, i].set_xlim(min_x_col - buffer, max_x_col + buffer)
        
        # Adjust the spacing between y-axis tick marks
        axs[rowIDX, i].set_yticks(axs[rowIDX, i].get_yticks()[::2])  # Adjust the step size as needed
    
    if dataset_sizes[i] == 15:
        mySize = "Small"
    elif dataset_sizes[i] == 60:
        mySize = "Medium"
    elif dataset_sizes[i] == 150:
        mySize = "Large"
    
    # Set x-axis ticks at the bottom for the top row and at the top for the bottom row
    if rowIDX == 0:
        axs[rowIDX, i].xaxis.set_ticks_position('bottom')
    else:
        axs[rowIDX, i].xaxis.set_ticks_position('top')
    
    # Increase x-axis label font size and ensure at least 5 ticks
    axs[1,i].set_xlabel(f"{mySize} Dataset \n ({dataset_sizes[i]} Features)", fontsize=14)
    
    # Set the number of tick marks on the x axis
    axs[0,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    axs[1,i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Add padding between x-axis tick labels and the plot
    axs[1, i].tick_params(axis='x', pad=10, labelsize=12)

# Remove y-axis tick marks for the middle and far right subplots of each row
for i in range(2):
    for j in range(1, 3):
        axs[i, j].set_yticklabels([])
        
for i in range(3):
    axs[0, i].set_xticklabels([])
        
# Add Y Axis Titles and increase font size
axs[0, 0].set_ylabel(performance_metric_ylabel, fontsize=14)
axs[1, 0].set_ylabel(other_metric_ylabel, fontsize=14)

# Add a single legend for all subplots
axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
axs[1, 2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

for i in range(2):
    # Get handles and labels of legend items
    handles, labels = axs[i, 2].get_legend_handles_labels()
    
    # Reverse the order
    handles.reverse()
    labels.reverse()
    
    # Create a new legend with the reversed order
    axs[i, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.tight_layout()
plt.show()


'''
#############################
#############################
##### PI PLOTS BY MODEL #####
#############################
#############################
'''

# Create a figure with three subplots in a horizontal layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each subplot
axs[1].set_xlabel('Data Size')
axs[0].set_ylabel('Average Total Feats')
for i, quantile in enumerate(['Total Feats 10%Q', 'Total Feats 50%Q', 'Total Feats 90%Q']):
    
    titleSegments = [' '.join(quantile.split(' ')[0:-1]), quantile.split(' ')[-1]]
    axs[i].set_title(f"{titleSegments[0]} ({titleSegments[1].replace('Q', '')} Quantile)")
    
    for j, modelName in enumerate(uniqueModels):
        # Get Model Data
        filteredModelData = finalResults_TotalFeats[finalResults_TotalFeats['Model'] == modelName]
        
        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')
        
        # Plot the data for the current model
        axs[i].plot(filteredModelData['Feat Size'], filteredModelData[quantile], 
                    marker=modelMarkerMap[modelName], label=f'{modelName}', color=color)

# Add a single legend for all subplots
axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

