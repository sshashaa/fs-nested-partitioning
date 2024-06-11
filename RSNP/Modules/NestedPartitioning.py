# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:47:01 2024

@author: ephouser
"""

from PythonModules import *
from NP_SolutionSearch import *
from EstimatingPerformance import *
from GeneratingDatasets import *

####################################################################################
####################################################################################

############################
# Nested Partitioning Driver
def preparePartitioning(inputsSettings, inputsData, inputsParameters, inputsContainers, inputsScreenResults, inputsPartitioning, inputsSeeds):
    
    ############################
    # Unpack Partitioning Inputs
    npOrder, expM_SettingPerf, expN_SettingIZ = inputsPartitioning
  
    #########################
    # USE NESTED PARTITIONING
        
    ######################################
    # PARTITION REGION PERFORMANCE MEASURE
    for EXP_M in range(len(expM_SettingPerf)):  
        
        #####################################
        # PARTITION INDIFFERENCE ZONE SETTING
        for EXP_N in range(len(expN_SettingIZ)):
        
            ################
            # START RUN TIME
            startTime_Partition = getTimeNow()
            
            #############################
            # PREPARE PARTITIONING INPUTS
            inputsPartitioning = [npOrder, EXP_M, expM_SettingPerf, EXP_N, expN_SettingIZ]
            
            ##############################
            # INITIATE NESTED PARTITIONING
            resultsPartitioning = nestedPartitioning(inputsSettings, inputsData, inputsParameters, inputsContainers, inputsScreenResults, inputsPartitioning, inputsSeeds)
            
            ##############################################
            # PREPARE PARTITIONING RESULTS FOR EXPORTATION
            exportResults(resultsPartitioning, startTime_Partition)
                
    return resultsPartitioning


####################################################################################
####################################################################################

##############################
# NESTED PARTITIONING FUNCTION
def nestedPartitioning(inputsSettings, inputsData, inputsParameters, inputsContainers, inputsScreenResults, inputsPartitioning, inputsSeeds):
    
    ###############
    # UNPACK INPUTS
    computer, INDIV, dataSize, EXP_I, EXP_J, EXP_J_STR, EXP_K = inputsSettings
    trueSol, dataX_All, dataX, dataY, dataX_MacJ, dataY_MacJ, dataTrainValidMic = inputsData
    iSR, seedTrainTestMic = inputsParameters 
    time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict = inputsContainers
    npOrder, EXP_M, expM_SettingPerf, EXP_N, expN_SettingIZ = inputsPartitioning
    seedSampleNodes, seedSurNodes, seedRepSample, seedIteration = inputsSeeds
    
    if EXP_K == 'NP':
        dataTrainValidMic = inputsScreenResults
        keptFeats = []
    else:
        EXP_L, expL_Screen, repScreening, screenTime, screenTimeMin, keptFeats, iSR, dataTrainValidMic, modelName = inputsScreenResults
    
    dataX_Train_MacJ, dataX_Test_MacJ = dataX_MacJ
    dataY_Train_MacJ, dataY_Test_MacJ = dataY_MacJ
    
    ###################
    # SET DATA INSTANCE
    dataInstance = dataSize+EXP_J_STR
    
    #####################
    # INITIALIZE COUNTERS
    k = 0                   # Tree Level
    featCount = 0           # Current Feature
    countNode = 0           # Current Node
    iSS = 0                 # Current Sample Count
    iSR = 0                 # Current Replication Count
                       
    d = len(npOrder['Variable'])                                             # Number of Variables
    n = max(20,math.floor(0.1*d))                                            # Number of Solutions to Sample
    r = 20   #20                                                             # Number of Replications
    settingSubRegionPerf = expM_SettingPerf[EXP_M]                           # Average = 0 | Minimum = 1 | Maximum = 2
    settingBacktrackType = 1                                                 # Surrounding Region: 0 | Solution Track = 1
    settingBacktrackNode = 1                                                 # 0: Best | 1: Lowest | 2: Highest
    settingIZ = expN_SettingIZ[EXP_N]                                        # 0: Static | 1: Deterministic Dynamic | 2: Dynamic(BT)
    deltaP = len(dataX.columns)/3                                            # Indifference Zone
    alpha = 0.05                                                             # Significance Level
    countBacktracks = []                                                     # Current Number of Backtracks
    NB = 1  # math.floor(d*0.1)                                              # Number of Iterations in Short-term Backtrack History
    
    inputsNPSettings = [settingSubRegionPerf, settingBacktrackType, settingBacktrackNode, settingIZ, countBacktracks]
    
    ###########
    # SET SEEDS
    random.seed(1)
    seedSolSample = random.choices(range(1,d), k=1000000)

    ########################
    # INFORMATION CONTAINERS
    pathFeats = []
    colNames = ["Level", "Node", "Partitioned Feat", "IoE", "Path Feats", "Path", "Sample Average", "BackTrack", "Branch Decision", "Parent", "Children", "Str Path", "Str Parent"]
    historyFullTree = pd.DataFrame(data=None, columns=colNames)
    colNames = ["Level", "Node", "Partitioned Feat", "IoE", "Path Feats", "Path", "Sample Average", "BackTrack", "Branch Decision", "Parent", "Children", "Str Path", "Str Parent", "Active"]
    historyCurrentTree = pd.DataFrame(data=None, columns=colNames)
    historySolPath = pd.DataFrame(data=None, columns=colNames)
    historySurrRegionNodes = pd.DataFrame(data=None, columns = colNames)
    historyReplications = []
    
    if settingSubRegionPerf == 0:
        YbarbarHistory = []     
    elif settingSubRegionPerf == 1:
        YbarMinHistory = []     
    elif settingSubRegionPerf == 2:
        YbarMaxHistory = []

    ####################
    # INCUMBENT SOLUTION
    XOPT = []
    ZOPT = [9999999999]
    count_NoImprov = 0
    
    print("\n")
    print("############################################")
    print("############ BEGIN PARTITIONING ############")
    print("############################################")
    print("\n")
    checked = 0
    count = 0
    k=0
    
    ###################
    # FOR EACH VARIABLE
    while k < len(npOrder['Variable']):
        
        ###########
        # UPDATE IZ
        if settingIZ == 1:
            deltaP = deltaP/1.5
        if settingIZ == 2:
            print("Need a way to decrease delta")
        
        ########################################
        # INITIALIZE LISTS TO STORE PERFORMANCES
        Yij = [[[]]*n, [[]]*n, [[]]*n]    
        tempRepHistory = []
        
        if settingSubRegionPerf == 0:       # AVERAGE
            Ybarbar = []
            Varbar = []
        elif settingSubRegionPerf == 1:     # MINIMUM
            YbarMin = []
            Varbar = []
        elif settingSubRegionPerf == 2:     # MAXIMUM
            YbarMax = []
            Varbar = []
        
        ###########################
        # GET TREE INFO FOR LEVEL K
        level = k
        partFeat = npOrder['Variable'].iloc[k]
        pathFeats = [npOrder['Variable'].loc[i] for i in range(k+1)]
        
        ##################################
        # IDENTIFY THE PARENT OF THIS NODE
        if k == 0:
            parent = []
        else:
            parent = list(historySolPath["Path"])[-1]
        
        # Adjust Sampling Rate
        # n = getUpdatedSamplingRate(n, countBacktracks, NB)
        
        #######################################################################
        
        ###################################
        ###################################
        ######### INCLUDE FEATURE #########
        ###################################
        ###################################
        
        #############################
        # GET TREE INFO FOR INCLUDING
        countNode += 1
        tempTreeInc = [level, countNode, partFeat, 1]    
        
        ###################
        # UPDATE CONTAINERS
        inputsContainers = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]
        
        ##################
        # SAMPLE SOLUTIONS
        random.seed(seedSampleNodes[countNode])  
        samplesInc, iSS = getSampleSolutions(1, npOrder, n, k, d, iSS, seedSolSample, inputsContainers)
        
        #############################################
        #             MICRO-REPLICATIONS            
        # SAMPLE REPLICATIONS AND MEASURE PERFORMANCE 
        random.seed(seedSampleNodes[countNode])  
        YbarInc, Yij[0], VarInc, iSR, tempR, dataTrainValidMic = getSamplePerformance(samplesInc, Yij[0], iSR, r, dataTrainValidMic, seedTrainTestMic, dataX, dataY, keptFeats, alpha, deltaP)    
        tempRepHistory.append(tempR) 
        
        ###################################################
        # SUBREGION ESTIMATED BY AVERAGE SAMPLE PERFORMANCE
        if settingSubRegionPerf == 0:
            Ybarbar.append(round((1/len(YbarInc))*sum(YbarInc),5))      
            Varbar.append(round((1/len(VarInc))*sum(VarInc),5))
    
        ###################################################
        # SUBREGION ESTIMATED BY MINIMUM SAMPLE PERFORMANCE
        elif settingSubRegionPerf == 1:
            YbarMin.append(round(min(YbarInc),6))                       
            Varbar.append(round((1/len(VarInc))*sum(VarInc),5))
    
        ###################################################
        # SUBREGION ESTIMATED BY MAXIMUM SAMPLE PERFORMANCE
        elif settingSubRegionPerf == 2:
            YbarMax.append(round(max(YbarInc),6))                       
            Varbar.append(round((1/len(VarInc))*sum(VarInc),5))
        
        #####################
        # STORE BEST SOLUTION
        zStar_Inc = min(YbarInc)
        xStar_Inc = samplesInc[YbarInc.index(zStar_Inc)]
        
        ###########################################
        # GENERATE THE PATH THAT LED TO THIS BRANCH
        if k>0:
            prevPath = list(historySolPath["Path"])[-1]        
            prevPath = prevPath.copy()
            prevPath.append(1)
        else:
            prevPath = [1]
            
        pathInc = prevPath
    
        ###################################################
        # IDENTIFY THE CHILDREN OF THE NEW 'INCLUDING' NODE
        if k == 0:
            childrenInc = [[1,1], [1,0]]
        else:
            childrenInc = [pathInc + [1], pathInc + [0]]
            
        #######################################################################   
        
        ###################################
        ###################################
        ######### EXCLUDE FEATURE #########
        ###################################
        ###################################
        
        ####################################
        # GET TREE INFORMATION FOR EXCLUDING
        countNode += 1
        tempTreeExc = [level, countNode, partFeat, 0]    
        
        ###################
        # UPDATE CONTAINERS
        inputsContainers = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications,time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]

        ##################
        # SAMPLE SOLUTIONS
        random.seed(seedSampleNodes[countNode])  
        samplesExc, iSS = getSampleSolutions(0, npOrder, n, k, d, iSS, seedSolSample, inputsContainers)
        
        #############################################
        #             MICRO-REPLICATIONS            
        # SAMPLE REPLICATIONS AND MEASURE PERFORMANCE 
        random.seed(seedSampleNodes[countNode])  
        YbarExc, Yij[1], VarExc, iSR, tempR, dataTrainValidMic = getSamplePerformance(samplesExc, Yij[1], iSR, r, dataTrainValidMic, seedTrainTestMic, dataX, dataY, keptFeats, alpha, deltaP)
        tempRepHistory.append(tempR) 
        
        ###################################################
        # SUBREGION ESTIMATED BY AVERAGE SAMPLE PERFORMANCE
        if settingSubRegionPerf == 0:
            Ybarbar.append(round((1/len(YbarExc))*sum(YbarExc),5))      
            Varbar.append(round((1/len(VarExc))*sum(VarExc),5))
    
        ###################################################
        # SUBREGION ESTIMATED BY MINIMUM SAMPLE PERFORMANCE
        elif settingSubRegionPerf == 1:
            YbarMin.append(round(min(YbarExc),6))                       
            Varbar.append(round((1/len(VarExc))*sum(VarExc),5))
    
        ###################################################
        # SUBREGION ESTIMATED BY MAXIMUM SAMPLE PERFORMANCE
        elif settingSubRegionPerf == 2:
            YbarMax.append(round(max(YbarExc),6))                       
            Varbar.append(round((1/len(VarExc))*sum(VarExc),5))    
                             
        #####################               
        # STORE BEST SOLUTION
        zStar_Exc = min(YbarExc)
        xStar_Exc = samplesInc[YbarExc.index(zStar_Exc)]

        ###########################################
        # GENERATE THE PATH THAT LED TO THIS BRANCH
        if k>0:
            prevPath = list(historySolPath["Path"])[-1]       
            prevPath = prevPath.copy()
            prevPath.append(0)
        else:
            prevPath = [0]
            
        pathExc = prevPath
        
        ###################################################
        # IDENTIFY THE CHILDREN OF THE NEW 'EXCLUDING' NODE
        if k == 0:
            childrenExc = [[0,1], [0,0]]
        else:
            childrenExc = [pathExc + [1], pathExc + [0]]
    
        ###############
        # STORE RESULTS
        perfData = {"YbarInc" : YbarInc, "VarInc" : VarInc,
                    "YbarExc" : YbarExc, "VarExc" : VarExc}
    
    
        #######################################################################
        
        if k > 0:
            ######################################
            ######################################
            ######### SURROUNDING REGION #########
            ######################################
            ######################################
            
            ###################
            # UPDATE CONTAINERS
            inputsContainers = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]

            ##################
            # SAMPLE SOLUTIONS
            random.seed(seedSurNodes[k])  
            samplesSur, iSS = getSurroundingRegion(npOrder, iSS, n, d, seedSolSample, inputsContainers)
            
            #############################################
            #             MICRO-REPLICATIONS            
            # SAMPLE REPLICATIONS AND MEASURE PERFORMANCE 
            random.seed(seedSampleNodes[countNode])  
            YbarSur, Yij[2], VarSur, iSR, tempR, dataTrainValidMic = getSamplePerformance(samplesSur, Yij[2], iSR, r, dataTrainValidMic, seedTrainTestMic, dataX, dataY, keptFeats, alpha, deltaP)
            tempRepHistory.append(tempR) 
            
            ###################################################
            # SUBREGION ESTIMATED BY AVERAGE SAMPLE PERFORMANCE
            if settingSubRegionPerf == 0:
                Ybarbar.append(round((1/len(YbarSur))*sum(YbarSur),5))      
                Varbar.append(round((1/len(VarSur))*sum(VarSur),5))
        
            ###################################################
            # SUBREGION ESTIMATED BY MINIMUM SAMPLE PERFORMANCE
            elif settingSubRegionPerf == 1:
                YbarMin.append(round(min(YbarSur),6))                       
                Varbar.append(round((1/len(VarSur))*sum(VarSur),5))
        
            ###################################################
            # SUBREGION ESTIMATED BY MAXIMUM SAMPLE PERFORMANCE
            elif settingSubRegionPerf == 2:
                YbarMax.append(round(max(YbarSur),6))                       
                Varbar.append(round((1/len(VarSur))*sum(VarSur),5))
            
            #####################               
            # STORE BEST SOLUTION
            zStar_Sur = min(YbarSur)
            xStar_Sur = samplesInc[YbarSur.index(zStar_Sur)]    
            
            ###############
            # STORE RESULTS
            perfData["YbarSur"], perfData["VarSur"] = YbarSur, VarSur
        
        
        ################
        # UPDATE VECTORS
        if k > 0:
            YbarInc, YbarExc, YbarSur = perfData["YbarInc"], perfData["YbarExc"], perfData["YbarSur"]
            VarInc, VarExc, VarSur = perfData["VarInc"], perfData["VarExc"], perfData["VarSur"]
        else:
            YbarInc, YbarExc = perfData["YbarInc"], perfData["YbarExc"]
            VarInc, VarExc = perfData["VarInc"], perfData["VarExc"]
        
        #################################
        # PRINT CURRENT VARIABLE POSITION
        print(f"Variable: {k+1}/{len(npOrder['Variable'])} | {EXP_I}.{EXP_J}")
        print("")
        
        ###############
        # PRINT RESULTS
        if settingSubRegionPerf == 0:               # AVERAGE
            
            #####################################################################
            # PRINT THE PERFORMANCE OF THE NEW NODES AND SURROUNDING REGION (K>0)
            print(npOrder['Variable'].iloc[k], " = 1 | Ybarbar = ", Ybarbar[0])
            print(npOrder['Variable'].iloc[k], " = 0 | Ybarbar = ", Ybarbar[1])    
            if k > 0: 
                print("Surr Region | Ybarbar = ", Ybarbar[2])
             
            YbarbarHistory.append(Ybarbar)
            YbarPerf = Ybarbar
        
        elif settingSubRegionPerf == 1:             # MINIMUM
            
            #####################################################################
            # PRINT THE PERFORMANCE OF THE NEW NODES AND SURROUNDING REGION (K>0)
            print(npOrder['Variable'].iloc[k], " = 1 | YbarMin = ", YbarMin[0])
            print(npOrder['Variable'].iloc[k], " = 0 | YbarMin = ", YbarMin[1])    
            if k > 0: 
                print("Surr Region | YbarMin = ", YbarMin[2])
             
            YbarMinHistory.append(YbarMin)
            YbarPerf = YbarMin
        
        elif settingSubRegionPerf == 2:             # MAXIMUM
            
            #####################################################################
            # PRINT THE PERFORMANCE OF THE NEW NODES AND SURROUNDING REGION (K>0)
            print(npOrder['Variable'].iloc[k], " = 1 | YbarMax = ", YbarMax[0])
            print(npOrder['Variable'].iloc[k], " = 0 | YbarMax = ", YbarMax[1])    
            if k > 0: 
                print("Surr Region | YbarMax = ", YbarMax[2])
             
            YbarMaxHistory.append(YbarMax)
            YbarPerf = YbarMax
        
        # #############################
        # # FORCE BACKTRACK FOR TESTING
        # if (k == 1) and (checked == 0):
        #     YbarPerf[2] = 0
        #     checked = 1
        # elif k == 3:
        #     YbarPerf[2] = 0
        
        #######################################################################
        
        ########################################            
        ########################################    
        ###### CHOOSE TO INCLUDE FEATURES ######
        ########################################
        ########################################
        
        if min(YbarPerf) == YbarPerf[0]:
            
            ##################################
            # UPDATE INCUMBENT (IF APPLICABLE)
            if YbarPerf[0] < ZOPT:
                ZOPT = YbarPerf[0]                    
                XOPT = [1]*len(keptFeats)+list(xStar_Inc)
                count_NoImprov = 0
            else:
                count_NoImprov += 1
            
            ##########################
            # UPDATE BACKTRACK TRACKER
            countBacktracks.append(0)
            
            ##############
            # UPDATE PATHS
            sampleAvgInc, sampleAvgExc = YbarPerf[0:2]
            backtrack, branchDecInc, branchDecExc = [0,1,0]
            
            tempTreeIncNew = [pathFeats, pathInc, sampleAvgInc, backtrack, branchDecInc, parent, childrenInc, (''.join(map(str, pathInc))), (''.join(map(str, parent)))]
            tempTreeExcNew = [pathFeats, pathExc, sampleAvgExc, backtrack, branchDecExc, parent, childrenExc, (''.join(map(str, pathExc))), (''.join(map(str, parent)))]
    
            for i in range(len(tempTreeIncNew)):            
                tempTreeInc.append(tempTreeIncNew[i])
                tempTreeExc.append(tempTreeExcNew[i])
            
            branchDec = 1
            k+=1
            
            ###################
            # UPDATE CONTAINERS
            inputsContainers = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]
            
            ################
            # UPDATE HISTORY
            historySolPath, historyFullTree, historyCurrentTree, historySurrRegionNodes = getUpdatedHistory(tempTreeInc, tempTreeExc, branchDec, inputsContainers)

        #######################################################################

        ########################################            
        ########################################    
        ###### CHOOSE TO EXCLUDE FEATURES ######
        ########################################
        ########################################   
          
        elif min(YbarPerf) == YbarPerf[1]:
                    
            ##################################
            # UPDATE INCUMBENT (IF APPLICABLE)
            if YbarPerf[1] < ZOPT:
                ZOPT = YbarPerf[1]
                XOPT = [1]*len(keptFeats)+list(xStar_Exc)
                count_NoImprov = 0
            else:
                count_NoImprov += 1
                
            ##########################
            # UPDATE BACKTRACK TRACKER
            countBacktracks.append(0)
            
            ##############
            # UPDATE PATHS
            sampleAvgInc, sampleAvgExc = YbarPerf[0:2]
            backtrack, branchDecInc, branchDecExc = [0,0,1]
            
            tempTreeIncNew = [pathFeats, pathInc, sampleAvgInc, backtrack, branchDecInc, parent, childrenInc, (''.join(map(str, pathInc))), (''.join(map(str, parent)))]
            tempTreeExcNew = [pathFeats, pathExc, sampleAvgExc, backtrack, branchDecExc, parent, childrenExc, (''.join(map(str, pathExc))), (''.join(map(str, parent)))]
    
            for i in range(len(tempTreeIncNew)):            
                tempTreeInc.append(tempTreeIncNew[i])
                tempTreeExc.append(tempTreeExcNew[i])
    
            branchDec = 0        
            k+=1
    
            ###################
            # UPDATE CONTAINERS
            inputsContainers = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]

            ################
            # UPDATE HISTORY
            historySolPath, historyFullTree, historyCurrentTree, historySurrRegionNodes = getUpdatedHistory(tempTreeInc, tempTreeExc, branchDec, inputsContainers)
            
        
        #######################################################################
        
        #################################            
        #################################    
        ###### CHOOSE TO BACKTRACK ######
        #################################
        #################################
        else:
            print(" ")
            print("BACKTRACKING REQUIRED.")
            
            ##################################
            # UPDATE INCUMBENT (IF APPLICABLE)
            if YbarPerf[2] < ZOPT:
                ZOPT = YbarPerf[2]
                XOPT = [1]*len(keptFeats)+list(xStar_Sur)
                count_NoImprov = 0
            else:
                count_NoImprov += 1
                
            ##########################
            # UPDATE BACKTRACK TRACKER
            countBacktracks.append(1)
            
            ##############
            # UPDATE PATHS
            sampleAvgInc, sampleAvgExc = YbarPerf[0:2]
            backtrack = 1
            branchDec = -1
            
            tempTreeIncNew = [pathFeats, pathInc, sampleAvgInc, backtrack, branchDec, parent, childrenInc, (''.join(map(str, pathInc))), (''.join(map(str, parent)))]
            tempTreeExcNew = [pathFeats, pathExc, sampleAvgExc, backtrack, branchDec, parent, childrenExc, (''.join(map(str, pathExc))), (''.join(map(str, parent)))]
    
            for i in range(len(tempTreeIncNew)):            
                tempTreeInc.append(tempTreeIncNew[i])
                tempTreeExc.append(tempTreeExcNew[i])
            
            ################################
            # IDENTIFY BACKTRACKING LOCATION
            historySolPath = getBacktrack(historySolPath, inputsNPSettings)
            
            if len(historySolPath) > 0:
                # Make sure full solution path is included
                firstLevel = min(historySolPath["Level"])
                while firstLevel > 0:
                    for parent in list(historySolPath["Parent"]):
                        if parent not in list(historySolPath["Path"]):
                            historySolPath = pd.concat([historySolPath,historyCurrentTree[historyCurrentTree['Path'].apply(lambda x: x == parent)]], ignore_index = True) 
                               
                    firstLevel = min(historySolPath["Level"])
                    historySolPath = historySolPath.sort_values(["Level", "Node"])
                    
                    # Reset the indices of the DataFrame
                    historySolPath = historySolPath.reset_index(drop=True)
                    
                # Reset the Level to the new node
                k = list(historySolPath["Level"])[-1] + 1
                
                # Make sure all parents are included in Surrounding Region  
                for parent in historySurrRegionNodes["Str Parent"]:
                    if parent not in historySurrRegionNodes["Str Path"]:
                        historySurrRegionNodes = pd.concat([historySurrRegionNodes,historyCurrentTree[historyCurrentTree['Str Path'].apply(lambda x: x == parent)]], ignore_index = True) 
                            
                historySurrRegionNodes = historySurrRegionNodes.sort_values(["Level", "Node"])
                
                # Reset the indices of the DataFrame
                historySurrRegionNodes = historySurrRegionNodes.reset_index(drop=True)
                
                # Update Containers
                inputsContainers = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]

                # Update History
                historySolPath, historyFullTree, historyCurrentTree, historySurrRegionNodes = getUpdatedHistory(tempTreeInc, tempTreeExc, branchDec, inputsContainers)
                
                print("Returning to Level", k-1,  " | Node", list(historySolPath["Node"])[-1])
    
            else:
                k = 0
                countNode = 0
                
                # Update Full Tree
                historyFullTree.loc[len(historyFullTree.index)] = tempTreeInc
                historyFullTree.loc[len(historyFullTree.index)] = tempTreeExc
    
                # Update the Surrounding Region
                historySurrRegionNodes = historySurrRegionNodes.drop(historySurrRegionNodes.index)            
    
                # Deactivate all Nodes in Current Tree
                for i in range(len(historyCurrentTree)):
                    historyCurrentTree['Active'].iloc[i] = 0
                    
                print("Returning to Level", k,  " | Node", 0)
                    
            
        historyReplications.append(tempRepHistory)
        # print("\n")
        # print(f"Current Solution Path: {historySolPath['Path Feats'].iloc[-1]}")    
        # print(f"                       {historySolPath['Path'].iloc[-1]}")

        if count_NoImprov >= 100:
            print("20 Iterations with no improvement")
            print(f"Optimal Solution = {XOPT}")
            print(f"Optimal Value = {ZOPT}")
            
            if EXP_K == "NP":
                modelName = "NP"
            elif EXP_K == 'RSP':
                modelName = modelName.replace("-", 'P-')
            
            tempTerminate = pd.DataFrame(data = [[modelName, dataInstance, k]], columns = ['Model', 'Data Instance', 'Last Itr'])
            
            k = len(npOrder['Variable'])
        
            #############################
            ### EXPORT TERMINATION RESULT    
            if computer == 0:
                termination_Path = "/home/ephouser/RSP/Results/TerminatedEarly.xlsx"
            else:
                termination_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\TerminatedEarly.xlsx"
            
            # Import Time Results
            if os.path.isfile(termination_Path):
                terminatedEarly = pd.read_excel(termination_Path, index_col=None)
            else:
                terminatedEarly = pd.DataFrame(data = None, columns = ["Model" , "Data Instance", "Last Itr"])
            
            # Add/Update Results DF
            if len(terminatedEarly) == 0:
                # Add row to the end of the dataframe
                terminatedEarly = pd.concat([terminatedEarly,tempTerminate], axis = 0, ignore_index=True)
                
            elif ((terminatedEarly['Model'] == tempTerminate.iloc[0,0]) & (terminatedEarly["Data Instance"] == tempTerminate.iloc[0,1])).any():
                # Replace the existing row
                terminatedEarly.loc[(terminatedEarly['Model'] == tempTerminate.iloc[0,0]) & (terminatedEarly['Data Instance'] == tempTerminate.iloc[0,1])] = tempTerminate.values
                
            else:
                terminatedEarly = pd.concat([terminatedEarly,tempTerminate], axis = 0, ignore_index=True)
                
            # Export Results
            terminatedEarly.to_excel(termination_Path, index = False)  
        
        elif backtrack == 0:
            print(" ")
            print("Branch Decision:", npOrder['Variable'].iloc[k-1], "=", branchDec)
            # print(f"{iSR} Replications Used")
            # print(f"Total Number of Backtracks = {sum(countBacktracks)}")
            # print(f"Backtracks in Last N Iterations = {sum(countBacktracks[NB:])}")
        
        print(f"Decisions Made: {k}/{len(npOrder['Variable'])}")
        print(f"Incumbent Solution: {XOPT}")
        print(f"Incumbent Objective: {ZOPT}")
        
        print("_____________________________________________")
        print("")
        
        count += 1
     
    # Compile
    inputsSettings = [computer, INDIV, dataSize, EXP_I, EXP_J, EXP_J_STR, EXP_K, modelName]
    inputsData = [trueSol, dataX_All, dataX, dataY, dataX_MacJ, dataY_MacJ, dataTrainValidMic]
    inputsParameters = [iSR, seedTrainTestMic]
    inputsContainers = [time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict]
    inputsPartitioning = [npOrder, EXP_M, expM_SettingPerf, EXP_N, expN_SettingIZ, countBacktracks]
    OutputsPartitioning = [pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, XOPT, ZOPT]
        
    if EXP_K == 'NP':
        inputsScreenResults = dataTrainValidMic
    else:
        inputsScreenResults = [EXP_L, expL_Screen, repScreening, screenTime, screenTimeMin, keptFeats, npOrder, iSR, dataTrainValidMic]
    
    resultsPartitioning = [inputsSettings, inputsNPSettings, inputsData, inputsParameters, inputsContainers, inputsScreenResults, inputsPartitioning, OutputsPartitioning]
    
    return resultsPartitioning


####################################################################################
####################################################################################

# Export Results
def exportResults(resultsPartitioning, startTime_Partition):
 
    # Unpack Containers
    inputsSettings, inputsNPSettings, inputsData, inputsParameters, inputsContainers, inputsScreenResults, inputsPartitioning, OutputsPartitioning = resultsPartitioning
    settingSubRegionPerf, settingBacktrackType, settingBacktrackNode, settingIZ, countBacktracks = inputsNPSettings            
    computer, INDIV, dataSize, EXP_I, EXP_J, EXP_J_STR, EXP_K, modelName = inputsSettings
    trueSol, dataX_All, dataX, dataY, dataX_MacJ, dataY_MacJ, dataTrainValidMic = inputsData
    iSR, seedTrainTestMic = inputsParameters
    time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict = inputsContainers
    
    npOrder, EXP_M, expM_SettingPerf, EXP_N, expN_SettingIZ, countBacktracks = inputsPartitioning
    pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, XOPT, ZOPT = OutputsPartitioning

    if EXP_K == 'NP':
        dataTrainValidMic = inputsScreenResults 
    else:
        EXP_L, expL_Screen, repScreening, screenTime, screenTimeMin, keptFeats, npOrder, iSR, dataTrainValidMic = inputsScreenResults

    ##########################
    ## Get Experiment Names ##
    ##########################
        
    # Get Data Instance
    dataInstance = dataSize+EXP_J_STR
        
    # Get Performance Type
    if settingSubRegionPerf == 0:
        perfType = "Avg"
    elif settingSubRegionPerf == 1:
        perfType = "Min"
    else:
        perfType = "Max"
    
    # Get IZ Setting
    if settingIZ == 0:
        izType = "SIZ"
    elif settingIZ == 1:
        izType = "DIZ"
    else:
        izType = "DBTIZ"
        
    # Determine Model
    if EXP_K == 'RSP':
        modelName = modelName.replace("-", 'P-')
        modelName = modelName + '-' + perfType + '-' + izType
        totalReps = iSR+repScreening[EXP_L]
    elif EXP_K == 'NP':
        modelName = "NP-"+perfType+'-'+izType
        totalReps = iSR
    
    ##################################################################################################
    ##################################################################################################
    
    ##############################
    ##############################
    ###  COMPILE TIME RESULTS  ###
    ##############################
    ##############################
    
    #####################
    ## GET FINAL RUN TIME
    endTime = getTimeNow()
    
    #######################################################
    # CALCULATE RUN TIME (CONSIDERING POSSIBLE DATE CHANGE)
    if endTime < startTime_Partition:
        # Handle the date change by adding a day to the end_time
        endTime += timedelta(days=1)

    ###############################
    # STORE TOTAL PARTITIONING TIME    
    partitionTime = endTime-startTime_Partition
    partitionTimeMin = round(partitionTime.total_seconds() / 60, 4)

    #########################
    # COMPILE TIME MILESTONES            
    if EXP_K == 'NP':
        totalTime = +partitionTime
        totalTimeMin = round(partitionTimeMin,4)
        timeSummary = [modelName, dataInstance, 0, partitionTimeMin, totalTimeMin]
        timeSummary = pd.DataFrame(data = [timeSummary], columns = ["Model" , "Data Instance", "Screen Time", "Partition Time", "Total Time"])

    else:
        totalTime = screenTime[EXP_L]+partitionTime
        totalTimeMin = round(screenTimeMin[EXP_L]+partitionTimeMin,4)
        timeSummary = [modelName, dataInstance, round(screenTimeMin[EXP_L],4), partitionTimeMin, totalTimeMin]
        timeSummary = pd.DataFrame(data = [timeSummary], columns = ["Model" , "Data Instance", "Screen Time", "Partition Time", "Total Time"])

    print("TIME RESULTS:")
    print(f"Total Run Time = {totalTime}")
    print(f"Total Run Time = {totalTimeMin} minutes")
    print("")
    

    #######################
    ### EXPORT TIME RESULTS    
    if computer == 0:
        resultsTime_Path = "/home/ephouser/RSP/Results/RSP_TimeBreakdown.xlsx"
    else:
        resultsTime_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\RSP_TimeBreakdown.xlsx"
    
    # Import Time Results
    if os.path.isfile(resultsTime_Path):
        resultsTime = pd.read_excel(resultsTime_Path, index_col=None)
    else:
        resultsTime = pd.DataFrame(data = None, columns = ["Model" , "Data Instance", "Screen Time", "Partition Time", "Total Time"])
    
    # Add/Update Results DF
    if len(resultsTime) == 0:
        # Add row to the end of the dataframe
        resultsTime = pd.concat([resultsTime,timeSummary], axis = 0, ignore_index=True)
        
    elif ((resultsTime['Model'] == timeSummary.iloc[0,0]) & (resultsTime["Data Instance"] == timeSummary.iloc[0,1])).any():
        # Replace the existing row
        resultsTime.loc[(resultsTime['Model'] == timeSummary.iloc[0,0]) & (resultsTime['Data Instance'] == timeSummary.iloc[0,1])] = timeSummary.values
        
    else:
        resultsTime = pd.concat([resultsTime,timeSummary], axis = 0, ignore_index=True)
        
    # Export Results
    resultsTime.to_excel(resultsTime_Path, index = False)  
    
    
    ##################################################################################################
    ##################################################################################################
    
    ######################################
    ######################################
    ###  COMPILE PARTITIONING RESULTS  ###
    ######################################
    ######################################
        
    ################
    # FINAL SOLUTION
    finalFeats = list(dataX.columns[np.array(XOPT)==1])
    
    ##############################
    # COMPILE PARTITIONING RESULTS
    colNames = ["Model", "Data Instance", "Solution", "Time"]
    tempPartitionResults = [modelName, dataInstance, finalFeats, totalTimeMin]
    partitionResultsDF = pd.DataFrame(data = [tempPartitionResults], columns = colNames)
    
    countTrue = 0
    tempNP_SOL = partitionResultsDF['Solution'].iloc[0]
    for xI in range(len(tempNP_SOL)):
        if 'x' in tempNP_SOL[xI]:
            countTrue += 1
            
    partitionResultsDF['Percent Correct'] = round(countTrue / len(trueSol),3)
    partitionResultsDF['Total Feats'] = len(tempNP_SOL)
    partitionResultsDF['Replications'] = totalReps

    ########################
    ###  EXPORT TRUE RESULTS

    # Compile True Results
    colNames = ["Model", "Data Instance", "Solution", "Time", "Percent Correct", "Total Feats", "Replications"]
    tempTrueResults = ["TRUE", dataInstance, trueSol, 0, 1, len(trueSol), 0]
    trueResultsDF = pd.DataFrame(data = [tempTrueResults], columns = colNames)

    if computer == 0:
        resultsSummary_Path = f"/home/ephouser/RSP/Results/AllResultsSummary.xlsx"
    else:
        resultsSummary_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\AllResultsSummary.xlsx"

    # Import Results (If Existing)
    if os.path.isfile(resultsSummary_Path):
        resultsSummary = pd.read_excel(resultsSummary_Path, index_col=None)
    else:
        resultsSummary = pd.DataFrame(data = None, columns = ["Model", "Data Instance", "Solution", "Time", "Percent Correct", "Total Feats", "Replications"])
                    
    # Add/Update Results DF
    if len(resultsSummary) == 0:
        # Add row to the end of the dataframe
        resultsSummary = pd.concat([resultsSummary,trueResultsDF], axis = 0, ignore_index=True)
        
    elif ((resultsSummary['Model'] == trueResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == trueResultsDF.iloc[0,1])).any():
        # Replace the existing row
        resultsSummary.loc[(resultsSummary['Model'] == trueResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == trueResultsDF.iloc[0,1])] = trueResultsDF.values
        

    else:
        resultsSummary = pd.concat([resultsSummary,trueResultsDF], axis = 0, ignore_index=True)
        
    # Export Results
    resultsSummary.to_excel(resultsSummary_Path, index = False)
    
    ################################
    ###  EXPORT PARTITIONING RESULTS
    
    # Import Results (If Existing)
    if os.path.isfile(resultsSummary_Path):
        resultsSummary = pd.read_excel(resultsSummary_Path, index_col=None)
    else:
        resultsSummary = pd.DataFrame(data = None, columns = ["Model", "Data Instance", "Solution", "Time", "Percent Correct", "Total Feats", "Replications"])
                    
    # Add/Update Results DF
    if len(resultsSummary) == 0:
        # Add row to the end of the dataframe
        resultsSummary = pd.concat([resultsSummary,partitionResultsDF], axis = 0, ignore_index=True)
        
    elif ((resultsSummary['Model'] == partitionResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == partitionResultsDF.iloc[0,1])).any():
        # Replace the existing row
        resultsSummary.loc[(resultsSummary['Model'] == partitionResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == partitionResultsDF.iloc[0,1])] = partitionResultsDF.values
        
    else:
        resultsSummary = pd.concat([resultsSummary,partitionResultsDF], axis = 0, ignore_index=True)
        
    # Export Results
    resultsSummary.to_excel(resultsSummary_Path, index = False)
    
    # Export Data
    if computer == 0:
        if INDIV == 1:
            historyCurrentTree.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historyCurrentTree_INDIV_{modelName}_{dataInstance}.xlsx")
            historyFullTree.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historyFullTree_INDIV_{modelName}_{dataInstance}.xlsx")
            historySolPath.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historySolPath_INDIV_{modelName}_{dataInstance}.xlsx")
            historySurrRegionNodes.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historySR_INDIV_{modelName}_{dataInstance}.xlsx")
            backtrackSummary.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/backTrack_INDIV_{modelName}_{dataInstance}.xlsx")
        else:
            historyCurrentTree.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historyCurrentTree_{modelName}_{dataInstance}.xlsx")
            historyFullTree.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historyFullTree_{modelName}_{dataInstance}.xlsx")
            historySolPath.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historySolPath_{modelName}_{dataInstance}.xlsx")
            historySurrRegionNodes.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/historySR_{modelName}_{dataInstance}.xlsx")
            backtrackSummary.to_excel(f"/home/ephouser/RSP/Results/TreeInfo/{dataSize}/backTrack_{modelName}_{dataInstance}.xlsx")
    
    else:
        
        tempFilePath = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\TreeInfo\\{dataSize}\\"
        if INDIV == 1:
            historyCurrentTree.to_excel(f"{tempFilePath}historyCurrentTree_INDIV_{modelName}_{dataInstance}.xlsx")
            historyFullTree.to_excel(f"{tempFilePath}historyFullTree_INDIV_{modelName}_{dataInstance}.xlsx")
            historySolPath.to_excel(f"{tempFilePath}historySolPath_INDIV_{modelName}_{dataInstance}.xlsx")
            historySurrRegionNodes.to_excel(f"{tempFilePath}historySR_INDIV_{modelName}_{dataInstance}.xlsx")
            backtrackSummary.to_excel(f"{tempFilePath}backTrack_INDIV_{modelName}_{dataInstance}.xlsx")
        else:
            historyCurrentTree.to_excel(f"{tempFilePath}historyCurrentTree_{modelName}_{dataInstance}.xlsx")
            historyFullTree.to_excel(f"{tempFilePath}historyFullTree_{modelName}_{dataInstance}.xlsx")
            historySolPath.to_excel(f"{tempFilePath}historySolPath_{modelName}_{dataInstance}.xlsx")
            historySurrRegionNodes.to_excel(f"{tempFilePath}historySR_{modelName}_{dataInstance}.xlsx")
            backtrackSummary.to_excel(f"{tempFilePath}backTrack_{modelName}_{dataInstance}.xlsx")
       

    ###################
    ## PRINT RESULTS ##  
    if EXP_K == 'RSP':
        try:
            print(f"Final Solution Performance: {historySolPath['Sample Average'].iloc[-1]}")
            print(f"Final Solution: {finalFeats}")    
            print(" ")
            print(f"Screening Replications Used = {repScreening[EXP_L]}")
            print(f"Partitioning Replications Used = {iSR}")
            print(f"Total Replications Used = {iSR+repScreening[EXP_L]}")
            print(" ")
            print(f"Total Number of Backtracks = {sum(countBacktracks)}")
        except:
            print("EMPTY DF")

    elif EXP_K == 'NP':
        try:
            print(f"Final Solution Performance: {historySolPath['Sample Average'].iloc[-1]}")
            print(f"Final Solution: {finalFeats}")    
            print(" ")
            print(f"Total Replications Used = {iSR}")
            print(" ")
            print(f"Total Number of Backtracks = {sum(countBacktracks)}")
        except:
            print("EMPTY DF")













