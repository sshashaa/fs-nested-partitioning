# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:35:24 2023

@author: ephouser
"""

import pandas as pd
from itertools import combinations
import random
import math

data = pd.read_excel("C:\\Users\\ephouser\\Downloads\\ScreeningData.xlsx")

#Number of Features
d=len(data.columns[1:])


alpha = 0.5   #Confidence Level
delta = 1   #Indifference Zone Parameter
b=[10]   #b(r): Number of new solutions in the rth iteration
R = 5   #R: Number of screening Iterations
ITilda = []   #I_Tilda(r): Indices of the b(r) solutions
IPrime=[]   #Set of Surviving Solutions through iteration r-1
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

def rapidScreening(data, inputs):
    
    for r in range(0,R+1):
        
        #Step 1 of Algorithm
        if r == 0:
            #Initial Solutions
            X.append(getInitialSolutions(d, b[0]))
            ITilda.append([i for i in range(0,b[0])])
            B.append(b[0])
            I.append(ITilda)
            Ni = [0 for i in ITilda]
            
        else:
            b.append(M-len(IPrime[r]))
            X.append(getNewSolutions_Shrinking())
            ITilda.append([B[r-1]+i for i in range(0,b[r])])
            B.append(B[r-1]+b[r])
            I.append(IPrime[r]+ITilda[r])
            Ni[r-1] = [0 for i in ITilda]
        
        #Step 2 of Algorithm
        for i in range(0,len(I[r])):
            #Compute Sample Average Ybari
            #Compute Screening Threshold Wil for all i,l, i!=l
            X
            
        #Step 3 of Algorithm
        #Pairwise comparison of all sample average estimates.
        #If Ybari - Ybarl <= -Wil, then remove solution i
        #Ranking tha remaining sample averages
            
            

    return
    
    
    
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
    
    return X
    
    
def getNewSolutions_Nearest(surSols, br):
    
    #Step 0: Rank solutions in IPrime by their index
    surSols.sort()
    b = 0
    
    while b < br:
        
        #If the m-neighborhood of Xi = Empty Set, then m+1 and check again
        #Once the above is false, generate one new solution uniformly from the m-neighborhood of Xi
        #Exclude this solution from X
        b
            
        
    return
 
def getNewSolutions_Shrinking(surSols, br, R, r):
    
    #Step 0: Rank solutions in IPrime by their index
    surSols.sort()
    b = 0
    m = R-r
    
    while b < br:
        
        #If the m-neighborhood of Xi = Empty Set, then m-1 and check again
        #Once the above is false, generate one new solution uniformly from the m-neighborhood of Xi
        #Exclude this solution from the sample space (so it cannot be visited later?)
        b
        
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    