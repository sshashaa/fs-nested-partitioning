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

data = pd.read_excel("C:\\Users\\ephouser\\Downloads\\ScreeningData_LR.xlsx")
dataX = data.iloc[:,1:]
dataY = data.iloc[:,0]

#Number of Features
d=len(data.columns[1:])

########################################
######  Parameter Initialization  ######
########################################

alpha = 0.5   #Confidence Level
delta = 1   #Indifference Zone Parameter
b=[10]   #b(r): Number of new solutions in the rth iteration
R = 5   #R: Number of screening Iterations
ITilda = [[]]   #I_Tilda(r): Indices of the b(r) solutions
IPrime=[[]]   #Set of Surviving Solutions through iteration r-1
I = []   #Union of surviving solutions and b(r) new solutions at the beginning of the rth iteration
B= []   #Cumulative number of solutions visited up to iteration r
n = []   #Number of replications allocated to each solution at iteration r (Round UP: n0*G**r)
n0 = 30   #Initial number of replications
G = 2   #Replication Growth Factor
r=0   #Current iteration
Ni= []   #Total number of observations on solution Xi up to iteration r for ever i in I(r)
Ybari = []   #Sample average
Wil = []   #Screening Threshold when comparing solutions Xi and Xl at iteration r
X = []   #Set of all solutions
M = 100 # User-specified integer constant 

random.seed(321)
seed_NewSol = random.sample(range(2**d), int((2**d)/10000))
s = 0

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
        for combos in combinations(range(d), m):
            Xj = X[0].copy()
            
            for i in combos:
                Xj[i] = 1 - Xj[i]
            Hm.append(Xj)
            
        #Uniformly select one solution from the m-neighborhood of X1 and designate it as X_2m+1
        random.seed(1)
        newX = Hm[random.randrange(len(Hm))]
        X.append(newX)
        
        #Flip all components of X_2m+1 (1->0 and 0->1) to yield X_2m+2. 
        newX = newX.copy()
        for i in range(len(newX)):
            if newX[i]==0:
                newX[i]=1
            else:
                newX[i]=0
        
        X.append(newX)
        
        b+=2
        m+=1
    
    return X[1:]

#Test
#Test = getInitialSolutions(d, b[0])
    
    
def getNewSolutions_Nearest(surSols, br):
    
    #Step 0: Rank solutions in IPrime by their index??
    b = 0
    genX = []
    
    for idx in surSols:
        
        
        ##### AM I DOING THIS RIGHT??? #####
        while b < br:
            m = 1
            emptyNeighborhood = True
            #Check if the m-neighborhood of Xi is empty
            while emptyNeighborhood == True:
                
                # Create the m-neighborhood of Xi
                Hm = []
                for combos in combinations(range(d), m):
                    Xi = X[idx].copy()
                    
                    for i in combos:
                        Xi[i] = 1 - Xi[i]
                    
                    if Xi not in X:
                        if Xi not in genX:
                            Hm.append(Xi)
                    
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
    m = R-r
    genX = []
    
    for idx in surSols:
        emptyNeighborhood = True
        #Check if the m-neighborhood of Xi is empty
        while emptyNeighborhood == True:
            
            # Create the m-neighborhood of Xi
            Hm = []
            for combos in combinations(range(d), m):
                Xi = X[idx].copy()
                
                for i in combos:
                    Xi[i] = 1 - Xi[i]
                
                if Xi not in surSols:
                    Hm.append(Xi)
            
            if len(Hm) == 0:
                m-= 1
            else:
                emptyNeighborhood = False
        
        #Uniformly select one solution from the m-neighborhood of Xi
        random.seed(seed_NewSol[s+b])
        newX = Hm[random.randrange(len(Hm))]
        b += 1
        genX.append(newX)
        
        if b == br:
            break
            
    return genX

#Test
#newNewTest = getNewSolutions_Shrinking(Test, M - len(IPrime[r]), 100, 5)
 
    
    
    
#the more a feature appears, the less likely we screen it
#after stopping, rank features by # of times they are selected in surviving solutions
#1.) Get rid of a lot of features
#2.) dont want to get rid of the good features
    
    
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
        tempITilda = [i for i in range(0,b[0]-1)]
        ITilda.append(tempITilda)
        
        #Update the cumulative num of solutions
        B.append(b[0])
        
        #Union of b(r) new solutions and IPrime surviving solutions
        I.append(tempITilda)
        
        #Initialize num of replications needed for each Xi
        Ni = [0 for i in range(len(tempITilda))]
        
       
    else:
        #Generate New Solutions
        b.append(M-len(IPrime[r]))
        s += b[r]           #Set new seeds for next iteration
        
        #Obtain New Solutions
        newSols = getNewSolutions_Nearest(IPrime[r], b[r])
        #newSols = getNewSolutions_Shrinking()
        
        X = X + newSols
        
        #Update I Tilda
        tempITilda = [B[r-1]+i for i in range(0,b[r])]
        ITilda.append(tempITilda)
        
        #Update cumulative num of solutions
        B.append(B[r-1]+b[r])
        
        #Union of b(r) new solutions and IPrime surviving solutions            
        I.append(IPrime[r-1]+tempITilda)
        
        #Initialize num of replications needed for each Xi
        Ni[r-1] = [0 for i in range(len(tempITilda))]
    
    #Step 2 of Algorithm (Approach B)
    Yij = []
    Ybari = []
    for i in range(0,len(I[r])):
        
        tempNewSol = np.array(newSols[i])
        
        #Take n(r) replications from Xi and set Ni(r)
        if r == 0:
            numReps = n0
        else:
            numReps = n[r]
            

        #Compute Sample Average Ybari
        #For Minimization Problems, multiply each Yij by -1***********
        tempYij = []
        for j in range(numReps):
            
            # Build Train and Test Datasets
            trainX, testX, trainY, testY = train_test_split(dataX.loc[:,tempNewSol==1], dataY, test_size = 0.2, random_state=seedsM[j])
            
            if modelType == 0:      # Linear Regression Model
                tempMAE, tempMSE, tempRMSE = fitLinearReg(trainX, testX, trainY, testY)
                tempYij.append(tempRMSE)
                
            elif modelType == 1:    # Logistic Regression Model
                f1_score, f2_score, f3_score, f4_score, precision, recall = fitLogReg(trainX, testX, trainY, testY)
                tempData = [f1_score, f2_score, f3_score, f4_score, precision, recall]
                tempYij.append(f3_score)

        Yij.append(np.array(tempYij)    )
        Ybari.append(round((1/numReps)*sum(tempYij),3))
        
    print(Yij)
    print(Ybari)
    
    
    #Compute Screening Threshold Wil for all i,l, i!=l
    #Wil = np.empty(len(I[r]),len(I[r]))
    Wil = np.empty((9,9))
    
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
                    Wil[i,l] = til*(Sil/math.sqrt(n0))
                else:
                    df = n[r]-1
                    til = scipy.stats.t.ppf(q = q, df = df)
                    Sil = math.sqrt((1/(n[r]-1)*sum((Yij[i]-Yij[l]-(Ybari[i]-Ybari[l]))**2)))
                    Wil[i,l] = round(til*(Sil/math.sqrt(n[r])),3)
    
    #Check to see which solutions should be removed
    #YbarDiff = np.empty(len(I[r])-1,len(I[r])-1)
    YbarDiff = np.empty((9,9))

    #WithinThreshold = np.empty(len(I[r])-1,len(I[r])-1)
    WithinThreshold = np.empty((9,9))
    
    ##########################################################################
    ############### THRESHOLD CHECK DEPENDS ON TYPE OF PROBLEM ###############
    ##########################################################################
    
    #### Max Objective Function (Log Reg: F Measure)
    #for i in range(len(I[r])):
    #    for l in range(len(I[r])):
    #        YbarDiff[i,l] = Ybari[i] - Ybari[l]
    #        
    #        if i == l:
    #            WithinThreshold = True
    #        elif YbarDiff[i,l] <= -1*Wil[i,l]:
    #            WithinThreshold[i,l] = False
    #        else:
    #            WithinThreshold[i,l] = True
                
    #### Min Objective Function (Lin Reg: RMSE)
    for i in range(len(I[r])):
        for l in range(len(I[r])):
            YbarDiff[i,l] = Ybari[i] - Ybari[l]
            
            if i == l:
                WithinThreshold[i,l] = True
            elif YbarDiff[i,l] <= Wil[i,l]:
                WithinThreshold[i,l] = True
            else:
                WithinThreshold[i,l] = False
    
    print(YbarDiff)
    print(WithinThreshold)
    
    tempIPrime = []
    for i in range(len(I[r])):
        if sum(WithinThreshold[i,:]) == len(I[r]):
            tempIPrime.append(i)
            
    IPrime.append(tempIPrime)
    
                    

                           
                
        
    #Step 3 of Algorithm
    #Pairwise comparison of all sample average estimates.
    #If Ybari - Ybarl <= -Wil, then remove solution i
    #Ranking tha remaining sample averages
    













