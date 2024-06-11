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
##################### COMPARING REPS MODELS ####################
#####################    FEATURE SELECTION   ####################
#################################################################
#################################################################
'''

# Create Label Categories
yLabels = {'RS20P-Hybrid_n1-Min-SIZ': "n₀ = 1", 
           'RS20P-Hybrid_n2-Min-SIZ': "n₀ = 2", 
           'RS20P-Hybrid_n3-Min-SIZ': "n₀ = 3",
           'RS20P-Hybrid_n4-Min-SIZ': "n₀ = 4",
           'RS20P-Hybrid_n5-Min-SIZ': "n₀ = 5",
           'RS20P-Hybrid-Min-SIZ': "n₀ = 10", }


# MODEL ORDER
uniqueModels = ['RS20P-Hybrid_n1-Min-SIZ','RS20P-Hybrid_n2-Min-SIZ', 
                'RS20P-Hybrid_n3-Min-SIZ', 'RS20P-Hybrid_n4-Min-SIZ', 
                'RS20P-Hybrid_n5-Min-SIZ', "RS20P-Hybrid-Min-SIZ"]
        
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
        featHeatMapRows.append(yLabels[myModel])
    
    
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
        
        # Sort Data
        featHeatMapValsDFSorted = featHeatMapValsDF[columns].T
        featHeatMapValsDFSorted = featHeatMapValsDFSorted.sort_values(by = [5, 4, 3, 2, 1, 0], ascending=False)
        featHeatMapValsDFSorted = featHeatMapValsDFSorted.T
        
        if i == 0:        
            sns.heatmap(featHeatMapValsDFSorted, cmap="Blues", ax=ax, cbar = False)
            ax.set_yticklabels(featHeatMapRows, rotation = 0, fontsize = 15)
            ax.set_xticks([])
            ax.set_xlabel(f"{category}", fontsize = 15)
            
        elif i == 1:
            sns.heatmap(featHeatMapValsDFSorted, cmap="Blues", ax=ax, cbar = False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{category}", fontsize = 15)
            
        elif i == 2:
            sns.heatmap(featHeatMapValsDFSorted, cmap="Blues", ax=ax, cbar = True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{category}", fontsize = 15)
    
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


    # Plot Title
    plt.suptitle(f'Heatmap of Selected Features \n ({dataSize} Data)', fontsize = 20)
    
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
# ##################### COMPARING REPS MODELS ####################
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




###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
    
'''
#################################################################
#################################################################
##################### COMPARING REPS MODELS ####################
######################      VIA MEDIAN      #####################
#################################################################
#################################################################
'''

# Plot Dimensions
width = 6
height = 5

# MODEL ORDER
uniqueModels = ['RS20P-Hybrid_n1-Min-SIZ', 'RS20P-Hybrid_n2-Min-SIZ', 
                'RS20P-Hybrid_n3-Min-SIZ', 'RS20P-Hybrid_n4-Min-SIZ', 
                'RS20P-Hybrid_n5-Min-SIZ', "RS20P-Hybrid-Min-SIZ"]
 
# COLOR MAP
modelColorMap = {"RS20P-Hybrid-Min-SIZ": 'darkred', 
                 'RS20P-Hybrid_n1-Min-SIZ': 'midnightblue', 
                 'RS20P-Hybrid_n2-Min-SIZ': 'royalblue',  
                 'RS20P-Hybrid_n3-Min-SIZ': 'lightskyblue',
                 'RS20P-Hybrid_n4-Min-SIZ': 'salmon',
                 'RS20P-Hybrid_n5-Min-SIZ': 'red',
                 }

# LABELS
modelLabelMap =  {'RS20P-Hybrid_n1-Min-SIZ': "n₀ = 1", 
                  'RS20P-Hybrid_n2-Min-SIZ': "n₀ = 2", 
                  'RS20P-Hybrid_n3-Min-SIZ': "n₀ = 3",
                  'RS20P-Hybrid_n4-Min-SIZ': "n₀ = 4",
                  'RS20P-Hybrid_n5-Min-SIZ': "n₀ = 5",
                  'RS20P-Hybrid-Min-SIZ': "n₀ = 10", }


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
    plt.title("Median Time (REPS Models)")
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
    plt.title("Median Percent Correct (REPS Models)")
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
    plt.title("Median # of Features (REPS Models)")
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
##################### COMPARING REPS MODELS ####################
######################      VIA AVERAGE     #####################
#################################################################
#################################################################
'''

# MODEL ORDER
uniqueModels = ['RS20P-Hybrid_n1-Min-SIZ', 'RS20P-Hybrid_n2-Min-SIZ', 
                'RS20P-Hybrid_n3-Min-SIZ', 'RS20P-Hybrid_n4-Min-SIZ', 
                'RS20P-Hybrid_n5-Min-SIZ', "RS20P-Hybrid-Min-SIZ"]
 
# COLOR MAP
modelColorMap = {"RS20P-Hybrid-Min-SIZ": 'darkred', 
                 'RS20P-Hybrid_n1-Min-SIZ': 'midnightblue', 
                 'RS20P-Hybrid_n2-Min-SIZ': 'royalblue',  
                 'RS20P-Hybrid_n3-Min-SIZ': 'lightskyblue',
                 'RS20P-Hybrid_n4-Min-SIZ': 'salmon',
                 'RS20P-Hybrid_n5-Min-SIZ': 'red',
                 }

# LABELS
modelLabelMap =  {'RS20P-Hybrid_n1-Min-SIZ': "$n_{pilot}$ = 1", 
                  'RS20P-Hybrid_n2-Min-SIZ': "$n_{pilot}$ = 2", 
                  'RS20P-Hybrid_n3-Min-SIZ': "$n_{pilot}$ = 3",
                  'RS20P-Hybrid_n4-Min-SIZ': "$n_{pilot}$ = 4",
                  'RS20P-Hybrid_n5-Min-SIZ': "$n_{pilot}$ = 5",
                  'RS20P-Hybrid-Min-SIZ': "$n_{pilot}$ = 10", }


#############################
#############################
##     BUILD TIME PLOTS    ##
##           REPS         ##
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
plt.title("Average Time (REPS)")
plt.xlabel('Number of Features')
plt.ylabel('Time (Seconds)')

# Reverse the order of handles and labels for legend
handles, labels = plt.gca().get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
plt.legend(handles, labels, loc='upper left')

plt.xticks(range(15, 151, 15), range(15, 151, 15), fontsize=14)
plt.yticks(fontsize=14)

# Adjust layout
plt.tight_layout()
plt.show()

################################
################################
## BUILD PERCENT CORRECT PLOT ##
##             REPS         ##
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
   
    # # Plot confidence intervals using the half-width column
    # plt.fill_between(list(filteredModelData['Feat Size']), 
    #                  list(filteredModelData['Percent Correct Avg'] - filteredModelData['Percent Correct 95% HW']), 
    #                  list(filteredModelData['Percent Correct Avg'] + filteredModelData['Percent Correct 95% HW']), 
    #                  color=color, alpha=0.05)  
   
    # Add labels and legend
    plt.title("Average Percent Correct (REPS)")
    plt.xlabel('Data Size')
    plt.ylabel('Percent Correct')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.xticks(range(15, 151, 15), range(15, 151, 15))


################################
################################
## BUILD TOTAL FEATURES PLOT  ##
##             REPS         ##
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

    # # Plot confidence intervals using the half-width column
    # plt.fill_between(list(filteredModelData['Feat Size']), 
    #                  list(filteredModelData['Total Feats Avg'] - filteredModelData['Total Feats 95% HW']), 
    #                  list(filteredModelData['Total Feats Avg'] + filteredModelData['Total Feats 95% HW']), 
    #                  color=color, alpha=0.3)  
    
# Add labels and legend outside the loop
plt.title("Average # of Features (REPS)")
plt.xlabel('Data Size')
plt.ylabel('# Features')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))   
plt.xticks(range(15, 151, 15), range(15, 151, 15))

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
##           REPS         ##
#############################
#############################

# General variables
performance_metric_dataset = finalResults_Time
performance_metric_name = "Time"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Size Labels
sizeLabels = {15: 'Small', 
              60: 'Medium', 
              150: 'Large'
                 }

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
    ax.set_xlabel(f'{sizeLabels[size]}', fontsize = 15)
    
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
##             REPS         ##
################################
################################

# General variables
performance_metric_dataset = finalResults_PercentCorrect
performance_metric_name = "Percent Correct"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Size Labels
sizeLabels = {15: '(True: 3 | Total: 15)', 
              60: '(True: 6 | Total: 60)', 
              150: '(True: 10 | Total: 150)'
                 }

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
    ax.set_xlabel(f'{sizeLabels[size]}', fontsize = 15)
    
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
##             REPS         ##
################################
################################     
    
# General variables
performance_metric_dataset = finalResults_TotalFeats
performance_metric_name = "Total Feats"
performance_metric_title = f"{performance_metric_name} (80% Prediction Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Size Labels
sizeLabels = {15: '15 Features', 
              60: '60 Features', 
              150: '150 Features'
                 }

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
    ax.set_xlabel(f'{sizeLabels[size]}', fontsize = 15)
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])
    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .8))

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
##           REPS         ##
#############################
#############################

# General variables
performance_metric_dataset = finalResults_Time
performance_metric_name = "Time"
performance_metric_title = f"{performance_metric_name} (95% Confidence Interval)"

# Size Labels
sizeLabels = {15: 'Small', 
              60: 'Medium', 
              150: 'Large'
                 }

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
    ax.set_xlabel(f'{sizeLabels[size]}', fontsize = 15)
    
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
##             REPS         ##
################################
################################

# General variables
performance_metric_dataset = finalResults_PercentCorrect
performance_metric_name = "Percent Correct"
performance_metric_title = f"Average Percent Correct \n 95% Confidence Interval"

# Define dataset sizes
dataset_sizes = [60, 150]

# Size Labels
sizeLabels = {60: '(True: 6 | Total: 60)', 
              150: '(True: 10 | Total: 150)'
                 }

# # Define dataset sizes
# dataset_sizes = [15, 60, 150]

# # Size Labels
# sizeLabels = {15: '(True: 3 | Total: 15)', 
#               60: '(True: 6 | Total: 60)', 
#               150: '(True: 10 | Total: 150)'
#                  }

# Create a figure with a single row and three columns of subplots
fig, axs = plt.subplots(1, len(dataset_sizes), figsize=(8, 4.5), sharey='row')

# Set title for the middle subplot of each row and increase font size
plt.suptitle(performance_metric_title, fontsize=20)

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

        # Get Bounds
        tempLB = max(filteredModelData[f'{performance_metric_name} Avg'].values[0] - filteredModelData[f'{performance_metric_name} 95% HW'].values[0],0)
        tempUB = min(filteredModelData[f'{performance_metric_name} Avg'].values[0] + filteredModelData[f'{performance_metric_name} 95% HW'].values[0],1)
        

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
        
        # Update Min and Max Vals
        LBvals.append(tempLB)
        UBvals.append(tempUB)
        

    # Add labels outside the loop
    ax.set_xlabel(f'{sizeLabels[size]}', fontsize = 16)
    
    # Remove y-axis tick labels
    ax.set_yticklabels([])

for ax in axs:
    minX = min(LBvals)
    maxX = max(UBvals)
    ax.set_xlim([minX-0.1, min(maxX+0.1,1)])  # Adjust the range as needed
    ax.set_xticks(np.arange(round(minX*5)/5 - 0.2, 1.2, 0.2))
    ax.set_xticklabels(ax.get_xticks(), fontsize = 16)
    ax.set_xticklabels([f'{tick:.1f}' for tick in ax.get_xticks()], fontsize=14)

    
# Place legend on the far right subplot
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1]  # Reverse the order of handles
labels = labels[::-1]    # Reverse the order of labels
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.03, .825), fontsize=14, frameon=False)

plt.tight_layout()
plt.show()  
  


################################
################################
## BUILD TOTAL FEATURES PLOT  ##
##             REPS         ##
################################
################################     
    
# General variables
performance_metric_dataset = finalResults_TotalFeats
performance_metric_name = "Total Feats"
performance_metric_title = f"{performance_metric_name} (95% Confidence Interval)"

# Define dataset sizes
dataset_sizes = [15, 60, 150]

# Size Labels
sizeLabels = {15: '15 Features', 
              60: '60 Features', 
              150: '150 Features'
                 }

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
    ax.set_xlabel(f'{sizeLabels[size]}', fontsize = 15)
    
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





