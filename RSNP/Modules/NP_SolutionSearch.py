# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:47:57 2024

@author: ephouser
"""

from PythonModules import *
from EstimatingPerformance import *
from GeneratingDatasets import *

################################
######  Define Functions  ######
################################

# Sample Solutions for getOrder
def getSampleSolutions(includeFeat, npOrder, n, k, d, iSS, seedSolSample, inputsContainers):

    pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict = inputsContainers
    
    sampleSet = []
    X0 = np.array([0 for i in range(len(npOrder['Variable']))])
    i=0
    
    while i < n:
        iSS +=1
        tempSol = X0.copy()
        
        if k > 0:
                       
            prevFeats = list(historySolPath["Path Feats"])[-1]
            prevPath = list(historySolPath["Path"])[-1]
                
            tempSol[k] = includeFeat
            idxUpdated = [k]
            
            for j in range(len(prevFeats)):
                npOrderFeats = npOrder['Variable']
                featIdx = npOrderFeats[npOrderFeats == prevFeats[j]].index[0]
                idxUpdated.append(featIdx)
                tempSol[featIdx] = prevPath[j]
                                
            idxRem = [idx for idx in range(d) if idx not in idxUpdated]
            posNewSol = []
            
            random.seed(iSS)
            tempIdx = random.choices(range(2), k=len(idxRem))
            
            for j in range(len(idxRem)):
               tempSol[idxRem[j]] = tempIdx[j]
           
            if sum(tempSol) > 0:
                sampleSet.append(tempSol)
                i+=1    

        else:
            random.seed(iSS)
            tempIdx = random.sample(range(d), seedSolSample[iSS])
            tempSol[tempIdx] = 1 - tempSol[tempIdx]
            tempSol[k] = includeFeat
            
            if sum(tempSol) > 0:
                sampleSet.append(tempSol)
                i+=1
            
    return sampleSet, iSS


# Sample Solutions for Surrounding Region
def getSurroundingRegion(npOrder, iSS, n, d, seedSolSample, inputsContainers):

    pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict = inputsContainers

    sampleSet = []
    X0 = np.array([0 for i in range(len(npOrder['Variable']))])
    i=0
    while i < n:
        iSS +=1
        tempSol = X0.copy()
        random.seed(iSS)
        tempIdx = random.sample(range(d), seedSolSample[iSS])
        tempSol[tempIdx] = 1 - tempSol[tempIdx]
        
        prevFeats = list(historySolPath["Path Feats"])[-1]
        prevPath = list(historySolPath["Path"])[-1]
                
        match = 0
        for j in range(len(prevPath)):
            npOrderFeats = npOrder['Variable']
            featIdx = npOrderFeats[npOrderFeats == prevFeats[j]].index[0]
            if tempSol[featIdx] == prevPath[j]:
                match +=1

        if match < len(prevPath):        
            if sum(tempSol) > 0:
                sampleSet.append(tempSol)
                i+=1  
            
    return sampleSet, iSS


# Sample Replications and Measure Performance WITH SUBSET SELECTION (Olaffson)
def getSamplePerformance(sampleSet, Yij, iSR, r, dataTrainTest, seedTrainTest, dataX, dataY, keptFeats, alpha, deltaP):
    
    #################################
    # Stage 1: Initial Replications #
    #################################
    Ybar = []
    Var = []
    repHistory = []
    for i in range(len(sampleSet)):
        #print(f"Calculating Performance for Solution {i}!")
        tempYij = Yij[i].copy()
        for j in range(r):
            iSR += 1
            
            trainX = dataTrainTest[j]['trainX']
            testX = dataTrainTest[j]['testX']
            trainY = dataTrainTest[j]['trainY']
            testY  = dataTrainTest[j]['testY']

            tempFullSample = np.array([1]*len(keptFeats)+list(sampleSet[i]))
            
            tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX.loc[:,tempFullSample==1],
                                                      testX.loc[:,tempFullSample==1],
                                                      trainY,
                                                      testY)
            tempYij.append(tempRMSE)
        
        tempYij = np.array(tempYij)
        tempYbar = round((1/len(tempYij))*sum(tempYij),5)
        tempVar = round(sum((tempYij-tempYbar)**2)/(r-1) ,5)
        
        Ybar.append(tempYbar)
        Var.append(tempVar)
        Yij[i] = tempYij.copy().tolist()

    ################################################
    # Stage 2: Filtering / Subset Selection        #
    ################################################
    
    q = 1-(alpha/2)
    df = r-1
    tVal = scipy.stats.t.ppf(q,df)

    W = np.zeros((len(sampleSet),len(sampleSet)))
    WThreshold = pd.DataFrame(data = np.zeros((len(sampleSet),len(sampleSet))))
    r_New = []
    for i in range(len(sampleSet)):
        for j in range(len(sampleSet)):
            if i != j:
                W[i][j] = tVal * math.sqrt((Var[i]+Var[j])/r)
                if (Ybar[i]-Ybar[j]) <= max(0,(W[i][j] - deltaP)):
                    WThreshold.iloc[i,j] = 1
                else:
                    WThreshold.iloc[i,j] = 0
            else:
                WThreshold.iloc[i,j] = 1
    
    for j in range(len(sampleSet)):
        if sum(WThreshold.iloc[:,j]) == len(WThreshold.iloc[:,j]):
            h = getRinotth(n = len(sampleSet), r0 = r, pstar = 1-alpha, conf = 0.99, rep = 10000)
            newTotalR = min(1000,math.ceil((h**2)*Var[j]/(deltaP**2)))
            # print(f"New Total R: {newTotalR} | Initial R = {r}")
            if newTotalR > r:
                repHistory.append(newTotalR)
                r_New.append(newTotalR-r)
            else:
                repHistory.append(r)
                r_New.append(0)
        else:
            # print(f"New Total R: 0| Initial R = {r}")
            repHistory.append(r)
            r_New.append(0)

    ####################################
    # Stage 2: Additional Replications #
    ####################################
    for i in range(len(sampleSet)):
        #print(f"Getting {r_New[i]} Additional Replications for Solution {i}!")
        
        if newTotalR > len(dataTrainTest):
            dataTrainTest = getMoreTrainTestData(dataX, dataY, 0.2, newTotalR - len(dataTrainTest), seedTrainTest, dataTrainTest)
        
        tempYij = list(Yij[i])
        for j in range(newTotalR):
            if j > (r-1):
                iSR += 1
            
                trainX = dataTrainTest[j]['trainX']
                testX = dataTrainTest[j]['testX']
                trainY = dataTrainTest[j]['trainY']
                testY  = dataTrainTest[j]['testY']

                tempFullSample = np.array([1]*len(keptFeats)+list(sampleSet[i]))
                tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX.loc[:,tempFullSample==1],
                                                          testX.loc[:,tempFullSample==1],
                                                          trainY,
                                                          testY)
                tempYij.append(tempRMSE)
    
        tempYij = np.array(tempYij)
        tempYbar = round((1/len(tempYij))*sum(tempYij),5)
        Ybar[i] = tempYbar
        Var[i] = round(sum((tempYij-tempYbar)**2)/(repHistory[i]-1) ,5)
        Yij[i] = tempYij.copy().tolist()
    
    # print('Node # ', countNode, ' | i = ', i, ' | delta = ', deltaP, '| R Total = ', r_Total, ' | iSR = ', iSR)
 
    return Ybar, Yij, Var, iSR, repHistory, dataTrainTest



# Reduce the Surrounding Region to ONLY Leaf Nodes (current ends of branches)
def reduceSurrRegion(surrRegionDF):
    # Remove Parent Nodes if Fully Represented by Child Nodes
    currentSurrPaths = list(surrRegionDF["Path"])
    for node in surrRegionDF["Node"]:
        childCount = 0
        for child in list(surrRegionDF.loc[surrRegionDF["Node"]==node,"Children"])[0]:
            if child in currentSurrPaths:
                childCount += 1
                
        if childCount == 2:
            surrRegionDF = surrRegionDF.drop(surrRegionDF[surrRegionDF["Node"] == node].index)

    return surrRegionDF



# Relocating to the best node in the Surrounding Region
def getBacktrack(historySolPath, inputsNPSettings):
    
    settingSubRegionPerf, settingBacktrackType, settingBacktrackNode, settingIZ, countBacktracks = inputsNPSettings
    
    # Backtrack to Surrounding Region
    if settingBacktrackType == 0:
        
        # Go to Best Node
        if settingBacktrackNode == 0:
            bestPerf = min(historySurrRegionNodes["Sample Average"])
            historySolPath = historySurrRegionNodes[historySurrRegionNodes["Sample Average"] == bestPerf].reset_index(drop=True)

        # Backtrack to lowest level in Surrounding Region
        elif settingBacktrackNode == 1:
            lowestLevel = max(historySurrRegionNodes["Level"])
            historySolPath = historySurrRegionNodes[historySurrRegionNodes["Level"] == lowestLevel].reset_index(drop=True)
            
            # Go to Best Node in Lowest Level (if multiple)
            if len(historySurrRegionNodes[historySurrRegionNodes["Level"] == lowestLevel])> 1:            
                bestPerf = min(historySolPath["Sample Average"])
                historySolPath = historySolPath[historySolPath["Sample Average"] == bestPerf].reset_index(drop=True)

        # Backtrack to Entire Feasible Region
        elif settingBacktrackNode == 2:
            historySolPath = historySolPath.iloc[:0]

    # Backtrack on Best Path
    elif settingBacktrackType == 1:
        
        # Go to Best Node
        if settingBacktrackNode == 0:
            if len(historySolPath["Sample Average"]) > 1:
                bestPerf = min(historySolPath["Sample Average"].iloc[:-1])
                historySolPath = historySolPath.iloc[:-1][historySolPath["Sample Average"] == bestPerf].reset_index(drop=True)
            else:
                historySolPath = historySolPath.iloc[:-1]
                
        # Backtrack to the Super Region
        elif settingBacktrackNode == 1:
            historySolPath = historySolPath.iloc[:-1]
                
        # Backtrack to the Entire Feasible Region
        elif settingBacktrackNode == 2:
            historySolPath = historySolPath.iloc[:0]
                
    return historySolPath



# Update the Tree
def getUpdatedHistory(newInc, newExc, branchDec, inputsContainers):

    pathFeats, historyFullTree, historyCurrentTree, historySolPath, historySurrRegionNodes, historyReplications, time_ALL_EXP, finalPartSolSummaryDF, backtrackSummary, finalSolDict, finalSelDict = inputsContainers
    
    # Update the Full Tree History
    newFullTree = historyFullTree.copy()
    newFullTree.loc[len(newFullTree.index)] = newInc
    newFullTree.loc[len(newFullTree.index)] = newExc
    
    # Set New Solution based on the Branch Decision
    if branchDec == 0:
        newSol = newExc
    elif branchDec == 1:
        newSol = newInc
    
    # Update the Solution Path   
    if branchDec != -1:
        newSol.append(1)
        newSolPath = historySolPath.copy()
        newSolPath.loc[len(newSolPath.index)] = newSol
    else:
        newSolPath = historySolPath.copy()

    # Update the Current Tree
    newCurrentTree = historyCurrentTree.copy()
    newPaths = newFullTree.iloc[-2:,:]
    newPaths['Active'] = [0,0]
    for i in range(len(newPaths)):
        strPath = newPaths['Str Path'].iloc[i]
        # Update or Add Node to Current Tree
        if strPath in newCurrentTree['Str Path'].values:
            newCurrentTree.loc[newCurrentTree.loc[:,'Str Path'] == strPath, :] = [newPaths.iloc[i]]
        else:
            newCurrentTree.loc[len(newCurrentTree.index)] = newPaths.iloc[i]
    
    # Update Node Activity
    for i in range(len(newCurrentTree['Str Path'])):
        if newCurrentTree['Str Path'].iloc[i] in newSolPath['Str Path'].values:
            newCurrentTree['Active'].iloc[i] = 1
        else:
            newCurrentTree['Active'].iloc[i] = 0            
    
    # Update the Surrounding Region
    newSurrReg = newCurrentTree[newCurrentTree['Active'] == 0]

    # Remove any branches after the current node
    currentLevel = newSolPath['Level'].iloc[-1]
    idxLevelTooHigh = newSurrReg[newSurrReg.loc[:,"Level"]>currentLevel].index
    newSurrReg = newSurrReg.drop(index = idxLevelTooHigh)
    
    # # Remove Parent Nodes if Fully Represented by Child Nodes
    # newSurrReg = reduceSurrRegion(newSurrReg)
    
    return newSolPath, newFullTree, newCurrentTree, newSurrReg



# Update Sampling Rate
def getUpdatedSamplingRate(n, countBacktracks, NB):
    
    if len(countBacktracks) >= NB:
        prevN_BT = sum(countBacktracks[NB:])
        total_BT = sum(countBacktracks)
        
        if (prevN_BT/NB) >= (total_BT/len(countBacktracks)):
            n += math.ceil(50 / len(countBacktracks))
        else:
            n -= math.ceil(50 / len(countBacktracks))
    
    
    return n



# Get Rinott's Constant h
def getRinotth(n, r0, pstar, conf, rep):
    # n = # of systems
    # r0 = first-stage sample size
    # pstar = 1-alpha value (PCS)
    # rep = # of replications to use for estimate
    # conf = confidence level on UB
    np.random.seed(1)

    Z = np.random.normal(size = (n-1)*rep).reshape(-1,n-1)
    Y = np.random.chisquare(n-1, size=(rep, n-1))
    C = np.random.chisquare(r0-1, size=(rep,1))
    Cmat = np.tile(C,(1,n-1))
    
    denom = np.sqrt((r0-1)*(1/Y+1/Cmat))
    H = np.sort(np.max(Z*denom,axis=1))
    Hstar = np.quantile(H,pstar)
    
    upper = int(np.ceil(pstar*rep+scipy.stats.norm.ppf(conf)*np.sqrt(pstar*(1-pstar)*rep) + 0.5))
    Hupper = H[upper]
    return Hupper