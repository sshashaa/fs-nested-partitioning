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
uniqueModels = ["RS20P-Hybrid-Min-SIZ", "GA", "NP-Min-SIZ",
                'ORS1-Shrinking', 'ORS2-Shrinking', 'ORS3-Shrinking', 'ORS20-Shrinking']

modelLabels = {"NP-Min-SIZ": 'NP',
                "ORS1-Shrinking": 'RS (1 Itr)', "ORS2-Shrinking": 'RS (2 Itr)',
                "ORS3-Shrinking": 'RS (3 Itr)',  "ORS20-Shrinking": 'RS (20 Itr)', 
                "GA": 'GA',      "RS20P-Hybrid-Min-SIZ": 'RSNP (20 Itr)', 
                 }
       
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
    featHeatMapRows = []
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
        featHeatMapRows.append(modelLabels[myModel])
        
    # Create Label Categories
    categories = {'True': [val for val in listFeats if 'x' in val], 
                  'Corr': [val for val in listFeats if 'corr' in val], 
                  'Noise': [val for val in listFeats if 'noise' in val]}
        
    # Set the figure size
    plt.figure(figsize=(18, 6))
    
    # Create a gridspec to specify the layout and relative sizes of subplots
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios)
    
    # Create subplots for each category
    for i, (category, columns) in enumerate(categories.items()):
        ax = plt.subplot(gs[i])
        
        # Sort Data
        featHeatMapValsDFSorted = featHeatMapValsDF[columns].T
        featHeatMapValsDFSorted = featHeatMapValsDFSorted.sort_values(by = [0,1,2,3,4,5,6], ascending=False)
        featHeatMapValsDFSorted = featHeatMapValsDFSorted.T
        
        if i == 0:        
            sns.heatmap(featHeatMapValsDFSorted, cmap="Blues", ax=ax, cbar = False)
            ax.set_yticklabels(featHeatMapRows, rotation = 0, fontsize = 24)
            ax.set_xticks([])
            ax.set_xlabel(f"{category}", fontsize = 24)
            
        elif i == 1:
            sns.heatmap(featHeatMapValsDFSorted, cmap="Blues", ax=ax, cbar = False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{category}", fontsize = 24)
            
        elif i == 2:
            sns.heatmap(featHeatMapValsDFSorted, cmap="Blues", ax=ax, cbar = False)
            ax.set_xticks([])
            ax.set_yticks([])
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



# ###################################################################################################################################################################################################
# ###################################################################################################################################################################################################
# ###################################################################################################################################################################################################
# ###################################################################################################################################################################################################
# ###################################################################################################################################################################################################

# '''
# #################################################################
# #################################################################
# ##################### COMPARING 20ITR MODELS ####################
# #####################    FEATURE SELECTION   ####################
# #################################################################
# #################################################################
# '''


# uniqueModels = finalResults['Model'].unique()
        
# for dataSize in finalResults['Data Size'].unique():
    
#     # Filter Dataset for Size
#     filteredDF = finalP2RESULTS[finalP2RESULTS["Data Size"] == dataSize]
    
#     # Get Features
#     if dataSize == "S":
#         listFeats = list(dataSmall.columns)
#     elif dataSize == "M":
#         listFeats = list(dataMedium.columns)
#     elif dataSize == "L":
#         listFeats = list(dataLarge.columns)
        
    
#     # Create DataFrame
#     featHeatMapValsDF = pd.DataFrame(data = None, columns = listFeats)
        
#     # Collect Data
#     for myModel in uniqueModels:
        
#         tempSols = filteredDF["Solution"][filteredDF["Model"] == myModel]
        
#         # Count Frequency of Selection for each Variable
#         varFreq = []
#         for var in listFeats:
#             varCount = 0
#             for sol in tempSols:
#                 if var in ast.literal_eval(sol):
#                     varCount += 1
                    
#             varFreq.append(varCount)
                    
        
#         tempDF = pd.DataFrame(data = [varFreq], columns = featHeatMapValsDF.columns)
        
#         featHeatMapValsDF = pd.concat([featHeatMapValsDF, tempDF], axis = 0, ignore_index = True)
#         featHeatMapValsDF = featHeatMapValsDF.apply(pd.to_numeric, errors='coerce')
#         featHeatMapRows = uniqueModels
    
    
#     # Create Label Categories
#     categories = {'Inf': [val for val in listFeats if 'x' in val], 
#                   'Corr': [val for val in listFeats if 'corr' in val], 
#                   'Noise': [val for val in listFeats if 'noise' in val]}

#     # Get tick positions and labels for each category
#     tick_positions = []
#     tick_labels = []
#     vline_positions = []
#     pos = 0
    
#     for category, columns in categories.items():
#         tick_positions.append(pos + len(columns) / 2)
#         tick_labels.append(category)
#         vline_positions.append(pos + len(columns))
#         pos += len(columns)
        
    
#     # Set the figure size
#     plt.figure(figsize=(10, 6))  # Adjust the width and height as needed
    
#     # Create a heatmap
#     sns.heatmap(featHeatMapValsDF, cmap="RdYlGn", yticklabels=featHeatMapRows)
    
#     # Set x-axis tick positions and labels
#     plt.xticks(tick_positions, tick_labels)
    
#     # Add dividing lines between categories
#     for position in vline_positions:
#         plt.axvline(position, color='black', lw=3)  # Adjust the color and line width as needed

#     # Set Plot Title
#     plt.title(f'Heatmap of Selected Features \n ({dataSize} Data)')
    
#     # Show the plot
#     plt.show()





###############################################################################
###############################################################################
##### GENERATE PLOTS
#####

# Plot Dimensions
width = 6
height = 5

# # Color Map
# modelColorMap = {'NP-Min-SIZ': 'black', 'ORS20-Hybrid': 'magenta', 'ORS20-Shrinking': 'lime',
#                  "RS1P-Hybrid-Min-SIZ": 'lightpink', "RS1P-Shrinking-Min-SIZ": 'deeppink',
#                  "RS2P-Hybrid-Min-SIZ": 'lightgreen', "RS2P-Shrinking-Min-SIZ": 'lime',
#                  "RS3P-Hybrid-Min-SIZ": 'lightblue', "RS3P-Shrinking-Min-SIZ": 'cyan', 
#                  "RS20P-Hybrid-Min-SIZ": 'plum', "RS20P-Shrinking-Min-SIZ": 'magenta', 
#                  }

# # MODEL ORDER
# uniqueModels = ['NP-Min-SIZ', 'ORS20-Hybrid', 'ORS20-Shrinking',
#                  "RS1P-Hybrid-Min-SIZ", "RS1P-Shrinking-Min-SIZ",
#                  "RS2P-Hybrid-Min-SIZ", "RS2P-Shrinking-Min-SIZ",
#                  "RS3P-Hybrid-Min-SIZ", "RS3P-Shrinking-Min-SIZ", 
#                  "RS20P-Hybrid-Min-SIZ", "RS20P-Shrinking-Min-SIZ"
#                  ]


###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
    
'''
#################################################################
#################################################################
##################### COMPARING 20ITR MODELS ####################
######################      VIA MEDIAN      #####################
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ['NP-Min-SIZ',
                'ORS1-Shrinking', 'ORS2-Shrinking', 'ORS3-Shrinking', 'ORS20-Shrinking', 
                'GA',
                "RS20P-Hybrid-Min-SIZ"]

# COLOR MAP
modelColorMap = {'NP-Min-SIZ': 'black', 'GA': 'darkviolet',
                 'ORS1-Shrinking': 'darkgreen', 'ORS2-Shrinking': 'lime', 
                 'ORS3-Shrinking': 'dodgerblue', 'ORS20-Shrinking': 'navy',
                 "RS20P-Hybrid-Min-SIZ": 'red'
                 }

# LABELS
modelLabelMap = {'NP-Min-SIZ': 'NP', 'GA': 'GA (Previous Work)',
                 'ORS1-Shrinking': 'RS1', 'ORS2-Shrinking': 'RS2', 'ORS3-Shrinking': 'RS3', 'ORS20-Shrinking': 'RS20',
                 "RS20P-Hybrid-Min-SIZ": 'RS20P-Hybrid (Ours)'
                 }


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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time 50%Q'], label=f'{modelLabelMap[modelName]}', color=color)
   
    # Add labels and legend
    plt.title("Median Time (20 Itr Models)")
    plt.xlabel('Data Size')
    plt.ylabel('Time')
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct 50%Q'], label=f'{modelLabelMap[modelName]}', color=color)
   
    # Add labels and legend
    plt.title("Median Percent Correct (20 Itr Models)")
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats 50%Q'], label=f'{modelLabelMap[modelName]}', color=color)
   
    # Add labels and legend
    plt.title("Median # of Features (20 Itr Models)")
    plt.xlabel('Data Size')
    plt.ylabel('# Features')
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss 50%Q'], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Median Percent Loss ")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Loss')
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
##################### COMPARING 20ITR MODELS ####################
######################      VIA AVERAGE     #####################
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ['NP-Min-SIZ',
                'ORS1-Shrinking', 'ORS2-Shrinking', 'ORS3-Shrinking', 'ORS20-Shrinking', 
                'GA',
                "RS20P-Hybrid-Min-SIZ"]

# COLOR MAP
modelColorMap = {'NP-Min-SIZ': 'black', 'GA': 'darkviolet',
                 'ORS1-Shrinking': 'darkgreen', 'ORS2-Shrinking': 'lime', 
                 'ORS3-Shrinking': 'dodgerblue', 'ORS20-Shrinking': 'navy',
                 "RS20P-Hybrid-Min-SIZ": 'red'
                 }

# LABELS
modelLabelMap = {'NP-Min-SIZ': 'NP', 'GA': 'GA (Previous Work)',
                 'ORS1-Shrinking': 'RS1', 'ORS2-Shrinking': 'RS2', 'ORS3-Shrinking': 'RS3', 'ORS20-Shrinking': 'RS20',
                 "RS20P-Hybrid-Min-SIZ": 'RSNP (Ours)'
                 }

#############################
#############################
##     BUILD TIME PLOTS    ##
##           20ITR         ##
#############################
#############################

# Initialize the plot
plt.figure(figsize=(6, 5))
for i in range(len(uniqueModels)):
       
    # Get Model Name
    modelName = uniqueModels[i] 
    
    # Get Model Data
    filteredModelData = finalResults_Time[finalResults_Time['Model'] == modelName]
    
    # Get Color Based on Model
    color = modelColorMap.get(modelName, 'black')
    
    # Plot a new line for each iteration with increased line width
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Time Avg'], label=f'{modelLabelMap[modelName]}', color=color, linewidth=3)
   
    # # Plot confidence intervals using the half-width column
    # plt.fill_between(list(filteredModelData['Feat Size']), 
    #                   list(filteredModelData['Time Avg'] - filteredModelData['Time 95% HW']), 
    #                   list(filteredModelData['Time Avg'] + filteredModelData['Time 95% HW']), 
    #                   color=color, alpha=0.05)  
    
# Add labels and legend outside the loop
plt.title("Average Time", fontsize=18)
plt.xlabel('Number of Features', fontsize=14)
plt.ylabel('Time (Min)', fontsize=14)

# Reverse the order of handles and labels for legend
handles, labels = plt.gca().get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
plt.legend(handles, labels, loc='upper left', fontsize=13, frameon=False)

plt.xticks(range(15, 151, 15), range(15, 151, 15), fontsize=14)
plt.yticks(fontsize=14)

# Adjust layout
plt.tight_layout()
plt.show()

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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Correct Avg'], label=f'{modelLabelMap[modelName]}', color=color)
   
    # Plot confidence intervals using the half-width column
    plt.fill_between(list(filteredModelData['Feat Size']), 
                     list(filteredModelData['Percent Correct Avg'] - filteredModelData['Percent Correct 95% HW']), 
                     list(filteredModelData['Percent Correct Avg'] + filteredModelData['Percent Correct 95% HW']), 
                     color=color, alpha=0.05)  
   
    # Add labels and legend
    plt.title("Average Percent Correct (20Itr)")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Correct')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


################################
################################
## BUILD TOTAL FEATURES PLOT  ##
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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Total Feats Avg'], label=f'{modelLabelMap[modelName]}', color=color)

    # Plot confidence intervals using the half-width column
    plt.fill_between(list(filteredModelData['Feat Size']), 
                     list(filteredModelData['Total Feats Avg'] - filteredModelData['Total Feats 95% HW']), 
                     list(filteredModelData['Total Feats Avg'] + filteredModelData['Total Feats 95% HW']), 
                     color=color, alpha=0.3)  
    
# Add labels and legend outside the loop
plt.title("Average # of Features (20Itr)")
plt.xlabel('Data Size')
plt.ylabel('# Features')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))   
plt.xticks(range(15, 151, 15), range(15, 151, 15))

plt.show()
    

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
    plt.plot(filteredModelData['Feat Size'], filteredModelData['Percent Loss Avg'], label=f'{modelName}', color=color)
   
    # Add labels and legend
    plt.title("Average Percent Loss ")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

#############################
#############################
##     BUILD TIME PLOTS    ##
##           20ITR         ##
#############################
#############################

# General variables
performance_metric_dataset = finalResults_Time
performance_metric_name = "Time"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(12, 4), sharey='row')

# Set title for the middle subplot of each row and increase font size
axs[1].set_title(performance_metric_title, fontsize=18)

# Loop over dataset sizes
for i, size in enumerate(dataset_sizes):
    ax = axs[i]  # Select the current subplot
        
    for j, modelName in enumerate(uniqueModels):

        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == size)]

        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')

        # Plot the prediction interval (10%-90% quantile)
        interval = ax.fill_betweenx([modelName]*len(filteredModelData), 
                                    filteredModelData[f'{performance_metric_name} 10%Q'], 
                                    filteredModelData[f'{performance_metric_name} 90%Q'], 
                                    alpha=1, color=color, label=f'{modelLabelMap[modelName]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        ax.plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        ax.plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)

    # Add labels outside the loop
    if i == 1:
        ax.set_xlabel(f'{performance_metric_name}')
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .8))

plt.tight_layout()
plt.show()  

    
   
################################
################################
## BUILD PERCENT CORRECT PLOT ##
##             20 ITR         ##
################################
################################

# General variables
performance_metric_dataset = finalResults_PercentCorrect
performance_metric_name = "Percent Correct"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(12, 4), sharey='row')

# Set title for the middle subplot of each row and increase font size
axs[1].set_title(performance_metric_title, fontsize=18)

# Loop over dataset sizes
for i, size in enumerate(dataset_sizes):
    ax = axs[i]  # Select the current subplot
        
    for j, modelName in enumerate(uniqueModels):

        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == size)]

        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')

        # Plot the prediction interval (10%-90% quantile)
        interval = ax.fill_betweenx([modelName]*len(filteredModelData), 
                                    filteredModelData[f'{performance_metric_name} 10%Q'], 
                                    filteredModelData[f'{performance_metric_name} 90%Q'], 
                                    alpha=1, color=color, label=f'{modelLabelMap[modelName]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        ax.plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        ax.plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)

    # Add labels outside the loop
    if i == 1:
        ax.set_xlabel(f'{performance_metric_name}')
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .8))

plt.tight_layout()
plt.show()    
  


################################
################################
## BUILD TOTAL FEATURES PLOT  ##
##             20 ITR         ##
################################
################################     
    
# General variables
performance_metric_dataset = finalResults_TotalFeats
performance_metric_name = "Total Feats"
performance_metric_title = f"80% Prediction Interval of Total Features (True Accuracy %)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(12, 4), sharey='row')

# Set title for the middle subplot of each row and increase font size
axs[1].set_title(performance_metric_title, fontsize=22)

LBvals = []
UBvals = []

# Loop over dataset sizes
for i, size in enumerate(dataset_sizes):
    ax = axs[i]  # Select the current subplot
        
    for j, modelName in enumerate(uniqueModels):

        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == size)]

        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')

        # Plot the prediction interval (10%-90% quantile)
        interval = ax.fill_betweenx([modelName]*len(filteredModelData), 
                                    filteredModelData[f'{performance_metric_name} 10%Q'], 
                                    filteredModelData[f'{performance_metric_name} 90%Q'], 
                                    alpha=1, color=color, label=f'{modelLabelMap[modelName]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        ax.plot(filteredModelData[f'{performance_metric_name} 10%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        ax.plot(filteredModelData[f'{performance_metric_name} 90%Q'], [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        
        LBvals.append(filteredModelData[f'{performance_metric_name} 10%Q'].values[0])
        UBvals.append(filteredModelData[f'{performance_metric_name} 90%Q'].values[0])
        
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    

for ax in axs:
    minX = min(LBvals)
    maxX = max(UBvals)
    # ax.set_xlim([minX-0.1, min(maxX+0.1,1)])  # Adjust the range as needed
    ax.set_xticks(np.arange(0, maxX, 20))
    ax.set_xticklabels(ax.get_xticks(), fontsize = 16)
    ax.set_xticklabels([f'{tick:.0f}' for tick in ax.get_xticks()], fontsize=16)
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.09, .625), frameon=False, fontsize = 14, markerfirst=False)

plt.tight_layout()
plt.show()  


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

#############################
#############################
##     BUILD TIME PLOTS    ##
##           20ITR         ##
#############################
#############################

# General variables
performance_metric_dataset = finalResults_Time
performance_metric_name = "Time"
performance_metric_title = f"{performance_metric_name} (95% Confidence Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(12, 4), sharey='row')

# Set title for the middle subplot of each row and increase font size
axs[1].set_title(performance_metric_title, fontsize=18)

# Loop over dataset sizes
for i, size in enumerate(dataset_sizes):
    ax = axs[i]  # Select the current subplot
        
    for j, modelName in enumerate(uniqueModels):

        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == size)]

        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')

        # Get Bounds
        tempLB = filteredModelData[f'{performance_metric_name} Avg'] - filteredModelData[f'{performance_metric_name} 95% HW']
        tempUB = filteredModelData[f'{performance_metric_name} Avg'] + filteredModelData[f'{performance_metric_name} 95% HW']

        # Plot the prediction interval (10%-90% quantile)
        interval = ax.fill_betweenx([modelName]*len(filteredModelData), 
                                    tempLB, 
                                    tempUB, 
                                    alpha=1, color=color, label=f'{modelLabelMap[modelName]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        ax.plot(tempLB, [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        ax.plot(tempUB, [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)

    # Add labels outside the loop
    if i == 1:
        ax.set_xlabel(f'{performance_metric_name}')
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .8))

plt.tight_layout()
plt.show()  

    
   
################################
################################
## BUILD PERCENT CORRECT PLOT ##
##             20 ITR         ##
################################
################################

# General variables
performance_metric_dataset = finalResults_PercentCorrect
performance_metric_name = "Percent Correct"
performance_metric_title = f"{performance_metric_name} (95% Confidence Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(12, 4), sharey='row')

# Set title for the middle subplot of each row and increase font size
axs[1].set_title(performance_metric_title, fontsize=18)

# Loop over dataset sizes
for i, size in enumerate(dataset_sizes):
    ax = axs[i]  # Select the current subplot
        
    for j, modelName in enumerate(uniqueModels):

        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == size)]

        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')

        # Get Bounds
        tempLB = filteredModelData[f'{performance_metric_name} Avg'] - filteredModelData[f'{performance_metric_name} 95% HW']
        tempUB = filteredModelData[f'{performance_metric_name} Avg'] + filteredModelData[f'{performance_metric_name} 95% HW']

        # Plot the prediction interval (10%-90% quantile)
        interval = ax.fill_betweenx([modelName]*len(filteredModelData), 
                                    tempLB, 
                                    tempUB, 
                                    alpha=1, color=color, label=f'{modelLabelMap[modelName]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        ax.plot(tempLB, [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        ax.plot(tempUB, [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        

    # Add labels outside the loop
    if i == 1:
        ax.set_xlabel(f'{performance_metric_name}')
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .8))

plt.tight_layout()
plt.show()  
  


################################
################################
## BUILD TOTAL FEATURES PLOT  ##
##             20 ITR         ##
################################
################################     
    
# General variables
performance_metric_dataset = finalResults_TotalFeats
performance_metric_name = "Total Feats"
performance_metric_title = f"{performance_metric_name} (95% Confidence Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(12, 4), sharey='row')

# Set title for the middle subplot of each row and increase font size
axs[1].set_title(performance_metric_title, fontsize=18)

# Loop over dataset sizes
for i, size in enumerate(dataset_sizes):
    ax = axs[i]  # Select the current subplot
        
    for j, modelName in enumerate(uniqueModels):

        # Get Model Data
        filteredModelData = performance_metric_dataset[(performance_metric_dataset['Model'] == modelName) & 
                                                        (performance_metric_dataset['Feat Size'] == size)]

        # Get Color Based on Model
        color = modelColorMap.get(modelName, 'black')

        # Get Bounds
        tempLB = filteredModelData[f'{performance_metric_name} Avg'] - filteredModelData[f'{performance_metric_name} 95% HW']
        tempUB = filteredModelData[f'{performance_metric_name} Avg'] + filteredModelData[f'{performance_metric_name} 95% HW']

        # Plot the prediction interval (10%-90% quantile)
        interval = ax.fill_betweenx([modelName]*len(filteredModelData), 
                                    tempLB, 
                                    tempUB, 
                                    alpha=1, color=color, label=f'{modelLabelMap[modelName]}')
        
        # Set linewidth for prediction interval lines
        interval.set_linewidth(3)

        # Add vertical markers on each data point with matching line color
        ax.plot(tempLB, [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)
        ax.plot(tempUB, [modelName]*len(filteredModelData), 'd', color=color, markersize=10, zorder=3)

    # Add labels outside the loop
    if i == 1:
        ax.set_xlabel(f'{performance_metric_name}')
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
    # Increase x-axis tick label size
    ax.tick_params(axis='x', labelsize=14)
    
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .8))

plt.tight_layout()
plt.show()  





