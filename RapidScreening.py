# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:35:24 2023

@author: ephouser
"""

import pandas as pd
from itertools import combinations
import numpy as np
import random, math, sys, os
sys.path.append('G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\')
from EstimatingPerformance import *
import scipy.stats
from datetime import datetime

startTime = datetime.now()

########################################
######  Parameter Initialization  ######
########################################

modelType = 0      #0 = Linear Regression, 1 = Logistic Regression
trainSize = 0.8             # Size of Training Set
testSize = round(1- trainSize,2)     # Size of Testing Set

alpha = 0.05   #Confidence Level
delta = 1   #Indifference Zone Parameter
b=[10]  #30    #b(r): Number of new solutions in the rth iteration
R = 4   #R: Number of screening Iterations
ITilda = [[]]   #I_Tilda(r): Indices of the b(r) solutions
IPrime=[[]]   #Set of Surviving Solutions through iteration r-1
I = []   #Union of surviving solutions and b(r) new solutions at the beginning of the rth iteration
B= []   #Cumulative number of solutions visited up to iteration r
n = []   #Number of replications allocated to each solution at iteration r (Round UP: n0*G**r)
n0 = 10   #Initial number of replications
G = 2   #Replication Growth Factor
r = 0   #Current iteration
Ni= []   #Total number of observations on solution Xi up to iteration r for ever i in I(r)
Ybari = []   #Sample average
Wil = []   #Screening Threshold when comparing solutions Xi and Xl at iteration r
X = []   #Set of all solutions
M = 5 #100  # User-specified integer constant 

random.seed(321)
seed_NewSol = random.sample(range(100000),100000) 
s = 0

random.seed(123)
seedsM = random.sample(range(100000),100000)      #Create CRN stream


########################################
########  Function Definitions  ########
########################################
   
def getInitialSolutions(d,b0):
    
    #Step 0
    X=[]
    b=2
    m=1
    
    #Create X1
    X.append([0]*d)
    #Create X2
    X.append([1]*d)

    
    #Step 1: Pick one solution uniformly from the m-neighborhood of X1 and designate it as X_2m+1.
    #Step 2: If b < b(0), return to Step 1. Otherwise, stop. Initial Solutions: [X1,X2,...,X_b(0)]
     
    while b < b0:
        
        #Create the m-neighborhood of X1
        Hm = []
        Hm = list(combinations(range(d), m))
            
        #Uniformly select one new solution from the m-neighborhood of X1 and designate it as X_2m+1
        
        for i0 in range(len(Hm)):
            random.seed(i0+1)
            tempSel = list(Hm[random.randrange(len(Hm))])
            
            Xj = np.array(X[0].copy())
            Xj[tempSel] = 1 - Xj[tempSel]
            
            if list(Xj) not in X:
                newX = Xj
                X.append(list(newX))
                break
        
        #Flip all components of X_2m+1 (1->0 and 0->1) to yield X_2m+2. 
        newX = newX.copy()
        newX = 1 - newX        
        X.append(list(newX))
        
        b+=2
        m+=1
    
    return X

#Test
#Test = getInitialSolutions(d, b[0])
    
    
def getNewSolutions_Nearest(surSols, br):
    
    #Step 0: Rank solutions in IPrime by their index??
    b = 0
    genX = []
    
    for idx in surSols:

        while b < br:
            m = 1
            emptyNeighborhood = True
            #Check if the m-neighborhood of Xi is empty
            while emptyNeighborhood == True:
                
                combos = list(combinations(range(d), m))
                
                # Create the m-neighborhood of Xi
                Hm = []
                
                Xi = np.array(X[idx].copy())
                for i0 in range(len(combos)):
                    
                    print("Itr R: ", r, "|", "Sol Index:", idx, "/", len(surSols), "| New Sol:", b, "/", br, "| m: ", m, "| ", "Combo", i0, "/", len(combos))
                    
                    c = list(combos[i0])
                    Xi[c] = 1 - Xi[c]
    
                    if list(Xi) not in X:
                        if list(Xi) not in genX:
                            Hm.append(list(Xi))
                    
                if len(Hm) == 0:
                    m+= 1
                else:
                    emptyNeighborhood = False
            
            #Uniformly select one solution from the m-neighborhood of Xi
            random.seed(seed_NewSol[s+b])
            newX = Hm[random.randrange(len(Hm))]
            b += 1
            genX.append(newX)
                        
    return genX

#Test
#newTest = getNewSolutions_Nearest(Test, M - len(IPrime[r]))
 
def getNewSolutions_Shrinking(surSols, br, R, r):
    
    #Step 0: Rank solutions in IPrime by their index??
    b = 0
    m = R-r+1
    genX = []
    
    for idx in surSols:
        while b < br:
            emptyNeighborhood = True
            #Check if the m-neighborhood of Xi is empty
            while emptyNeighborhood == True:
                
                combos = list(combinations(range(d), m))
                
                # Create the m-neighborhood of Xi
                Hm = []
                
                Xi = np.array(X[idx].copy())
                for i0 in range(len(combos)):                    
                    Xi[combos[i0]] = 1 - Xi[combos[i0]]
    
                    if list(Xi) not in X:
                        if list(Xi) not in genX:
                            Hm.append(list(Xi))
                
                if len(Hm) == 0:
                    m-= 1
                    if m==0:
                        emptyNeighborhood = False
                        b = br

                else:
                    emptyNeighborhood = False
            
                    #Uniformly select one solution from the m-neighborhood of Xi
                    random.seed(seed_NewSol[s+b])
                    newX = Hm[random.randrange(len(Hm))]
                    b += 1
                    genX.append(newX)
            
    return genX

#Test
#newNewTest = getNewSolutions_Shrinking(Test, M - len(IPrime[r]), 100, 5)
 
    
    
    
#the more a feature appears, the less likely we screen it
#after stopping, rank features by # of times they are selected in surviving solutions
#1.) Get rid of a lot of features
#2.) dont want to get rid of the good features
    
if modelType == 0:
        
    #data = pd.read_excel("C:\\Users\\ephouser\\Downloads\\ScreeningData_LR.xlsx")
    data = pd.read_csv("C:\\Users\\ethan\\Downloads\\GenData_LinReg.csv")
    dataX = data.iloc[:,:-1]
    dataY = data.iloc[:,-1]
    d=len(data.columns[:-1])
    
elif modelType == 1:
    #data = pd.read_excel("C:\\Users\\ephouser\\Downloads\\ScreeningData_LR.xlsx")
    data = pd.read_csv("C:\\Users\\ethan\\Downloads\\GenData_LogTest.csv")
    dataX = data.iloc[:,:-2]
    dataY = data.iloc[:,-1]
    d=len(data.columns[:-2])

#Number of Features

s = 0

for r in range(0,R+1):
    
    newSols = []        #Reset list of new solutions
    tempITilda = []     #Reset list of indices for new solutions
    n.append(n0*G**r)   #Number of replications allocated to each solution at iteration r
    #Step 1 of Algorithm
    if r == 0:
        #Obtain Initial Solutions
        newSols = getInitialSolutions(d, b[0])
        X = X + newSols
        
        #Update I Tilda
        tempITilda = [i for i in range(0,b[0])]
        ITilda.append(tempITilda)
        
        #Update the cumulative num of solutions
        B.append(b[0])
        
        #Union of b(r) new solutions and IPrime surviving solutions
        I.append(tempITilda)
        
        #Initialize num of replications needed for each Xi
        Ni = [0 for i in range(len(tempITilda))]
        
       
    else:
        #Generate New Solution
        b.append(M-len(IPrime[r]))
        s += b[r]           #Set new seeds for next iteration
        
        #Obtain New Solutions
        newSols = getNewSolutions_Nearest(IPrime[r], b[r])
        #newSols = getNewSolutions_Shrinking(IPrime[r], b[r], R, r)
        
        X = X + newSols
        
        #Update I Tilda
        tempITilda = [B[r-1]+i for i in range(0,len(newSols))]
        ITilda.append(tempITilda)
        
        #Update cumulative num of solutions
        B.append(B[r-1]+b[r])
        
        #Union of b(r) new solutions and IPrime surviving solutions            
        I.append(IPrime[r]+tempITilda)
        
        #Initialize num of replications needed for each Xi
        #Ni[r-1] = [0 for i in range(len(tempITilda))]
    
    #Step 2 of Algorithm (Approach B)
    Yij = []
    Ybari = []
    for i in I[r]:
        
        tempNewSol = np.array(X[i])
        
        #Includes the soltuion of no variables selected (All Zeros)
        if sum(tempNewSol) == 0:
            Yij.append(-1*99999)
            Ybari.append(-1*99999)
        else:
            
            #Take n(r) replications from Xi and set Ni(r)
            if r == 0:
                numReps = n0
            else:
                numReps = n[r]
            
            if modelType == 0:
                #Compute Sample Average Ybari
                #For Minimization Problems, multiply each Yij by -1***********
                tempYij = []
                for j in range(numReps):
                    
                    # Build Train and Test Datasets
                    trainX, testX, trainY, testY = train_test_split(dataX.loc[:,tempNewSol==1], dataY, test_size = 0.2, random_state=seedsM[j])
                    
                    tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX, testX, trainY, testY)
                    tempYij.append(-1*tempRMSE)
                    
        
                Yij.append(np.array(tempYij))
                Ybari.append(round((1/numReps)*sum(tempYij),3))
                
            elif modelType == 1:
                #Compute Sample Average Ybari
                tempYij = []
                for j in range(numReps):
                    
                    # Build Train and Test Datasets
                    trainX, testX, trainY, testY = train_test_split(dataX.loc[:,tempNewSol==1], dataY, test_size = 0.2, random_state=seedsM[j])
                    
                    #################################################
                    ## ERROR HERE FOR MULTINOMIAL LOG REG ROC CURVE##
                    #################################################
                    
                    f1_score, f2_score, f3_score, f4_score, precision, recall = fitLogReg(trainX, testX, trainY, testY)
                    tempData = [f1_score, f2_score, f3_score, f4_score, precision, recall]
                    tempYij.append(f3_score)
        
                Yij.append(np.array(tempYij))
                Ybari.append(round((1/numReps)*sum(tempYij),3))                
        
        
    #print(Yij)
    #print(Ybari)
    
    
    #Compute Screening Threshold Wil for all i,l, i!=l
    Wil = np.empty((len(I[r]),len(I[r])))
    
    for i in range(len(I[r])):
        for l in range(len(I[r])):
            if i == l:
                Wil[i,l] = 0.000
            else:
                
                q = 1-(alpha/(len(I[r])-1))
                
                if r == 0:
                    df = n0-1
                    til = scipy.stats.t.ppf(q = q, df = df)
                    Sil = math.sqrt((1/(n0-1)*sum((Yij[i]-Yij[l]-(Ybari[i]-Ybari[l]))**2)))
                    Wil[i,l] = round(til*(Sil/math.sqrt(n0)),3)
                else:
                    df = n[r]-1
                    til = scipy.stats.t.ppf(q = q, df = df)
                    Sil = math.sqrt((1/(n[r]-1)*sum((Yij[i]-Yij[l]-(Ybari[i]-Ybari[l]))**2)))
                    Wil[i,l] = round(til*(Sil/math.sqrt(n[r])),3)
    
    #Check to see which solutions should be removed
    YbarDiff = np.empty((len(I[r]),len(I[r])))
    WithinThreshold = np.empty((len(I[r]),len(I[r])))
    
    #############################################################################
    ############### THRESHOLD CHECK DEPENDS ON TYPE OF PROBLEM??? ###############
    #############################################################################
    
    for i in range(len(I[r])):
        for l in range(len(I[r])):
            YbarDiff[i,l] = Ybari[i] - Ybari[l]
            
            if i == l:
                WithinThreshold[i,l] = True
            elif YbarDiff[i,l] <= -1*Wil[i,l]:
                WithinThreshold[i,l] = False
            else:
                WithinThreshold[i,l] = True
    
    #print(YbarDiff)
    #print(WithinThreshold)
    
    tempIPrime = []
    for i in range(len(I[r])):
        if sum(WithinThreshold[i,:]) == len(I[r]):
            tempIPrime.append(i)
            
    IPrime.append(tempIPrime)
    
    

endTime = datetime.now()
totalTime = endTime-startTime


screenSol = []

for i in I[-1]:
    
    #if i > b[0]:
    screenSol.append(X[i])

screenDF = pd.DataFrame(screenSol)

tempSummary = []
for i in range(len(screenDF.columns)):
    count = sum(screenDF.iloc[:,i])
    tempSummary.append([dataX.columns[i] ,count])

names = ["Variable", "Times Selected"]
varSummary = pd.DataFrame(tempSummary, columns = names).sort_values(by=["Times Selected", "Variable"], ascending = False)
print(varSummary)
print("Total Run Time = ", totalTime)























