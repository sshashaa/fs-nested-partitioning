# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:30:06 2024

@author: ephouser
"""

from PythonModules import *

########################################
########  Function Definitions  ########
########################################

# Function to Generate Initial Solutions
def getInitialSolutions(d, b0, budgetLeft, seed_I0):
    
    #Step 0
    X=[]
    b=2
    m=1
    i0=0
    
    #Create X1
    X.append([0]*d)
    #Create X2
    X.append([1]*d)

    #Step 1: Pick one solution uniformly from the m-neighborhood of X1 and designate it as X_2m+1.
    #Step 2: If b < b(0), return to Step 1. Otherwise, stop. Initial Solutions: [X1,X2,...,X_b(0)]
     
    while b < b0:

        #Uniformly select one new solution from the m-neighborhood of X1 and designate it as X_2m+1
        uniqueSol = False
        while uniqueSol == False:
            i0 += 1
            random.seed(seed_I0[i0])
            tempSel = list(random.sample(range(d),m))            

            Xj = np.array(copy.deepcopy(X[0]))
            Xj[tempSel] = 1 - Xj[tempSel]
            
            if list(Xj) not in X:
                newX = Xj
                X.append(list(newX))
                uniqueSol = True
                break
        
        #Flip all components of X_2m+1 (1->0 and 0->1) to yield X_2m+2. 
        newX = copy.deepcopy(newX)
        newX = 1 - newX        
        X.append(list(newX))
        
        b+=2
        
        # If we have more neighborhoods than sampled solutions, then we find the step size
        if d > b0:
            m+= math.floor(d/b0)
        else:
            m += 1
            
        # If the new neighborhood does not exist, start over (only happens for small problems)
        if m >= d:
            m=2
            
    return X, budgetLeft

# Function to Generate New Solutions (Nearest Neighborhood)
def getNewSolutions_Nearest(surSols, br, i1, r, budgetLeft, d, E, X, seed_I1):

    #Step 0: Rank solutions in IPrime by their index??
    b = 0
    genX = []
    while b < br:
        for idx in surSols:
            m = 1
            emptyNeighborhood = True
            tempE = [X[e] for e in E]
            #Check if the m-neighborhood of Xi is empty
            while emptyNeighborhood == True:
                
                #Uniformly select one new solution from the m-neighborhood of Xi
                uniqueSol = False
                #print("d = ", d, " | m = ", m)
                HmSize = int((math.factorial(d))/((math.factorial(m))*(math.factorial(d-m))))
                HmCounter = 0
                while uniqueSol == False:
                    i1 += 1
                    HmCounter += 1
                    random.seed(seed_I1[i1])
                    tempSel = list(random.sample(range(d),m))            

                    Xi = np.array(copy.deepcopy(X[idx]))
                    Xi[tempSel] = 1 - Xi[tempSel]
               
                    if list(Xi) not in X:   
                        if list(Xi) not in genX:
                            if list(Xi) not in tempE:   # This line might be redundant
                                newX = list(Xi)
                                uniqueSol = True
                                break
                    
                    if HmCounter == HmSize:
                        break
                                                
                if uniqueSol == False:
                    m+= 1
                    if m > d:
                        print('All solutions exhausted.')
                        budgetLeft = False
                        return genX, budgetLeft
                else:
                    emptyNeighborhood = False
            
            b += 1
            genX.append(newX)
            
            if b == br:
                break
                        
    return genX, budgetLeft, i1

# Function to Generate New Solutions (Shrinking Neighborhood)
def getNewSolutions_Shrinking(surSols, br, i1, r, budgetLeft, d, E, X, seed_I1):
    
    #Step 0: Rank solutions in IPrime by their index??
    b = 0
    genX = []
    while b < br:
        for idx in surSols:
            m = d
            emptyNeighborhood = True
            tempE = [X[e] for e in E]
            #Check if the m-neighborhood of Xi is empty
            while emptyNeighborhood == True:
                
                #Uniformly select one new solution from the m-neighborhood of Xi
                uniqueSol = False
                #print("d = ", d, " | m = ", m)
                HmSize = int((math.factorial(d))/((math.factorial(m))*(math.factorial(d-m))))
                HmCounter = 0
                while uniqueSol == False:
                    i1 += 1
                    HmCounter += 1
                    random.seed(seed_I1[i1])
                    tempSel = list(random.sample(range(d),m))            

                    Xi = np.array(copy.deepcopy(X[idx]))
                    Xi[tempSel] = 1 - Xi[tempSel]
               
                    if list(Xi) not in X:
                        if list(Xi) not in genX:
                            if list(Xi) not in tempE:
                                newX = list(Xi)
                                uniqueSol = True
                                break
                    
                    if HmCounter == HmSize:
                        break
                                                
                if uniqueSol == False:
                    m-= 1
                    if m == 0:
                        print('All solutions exhausted. Possible Error.')
                        budgetLeft = False
                        return genX, budgetLeft
                else:
                    emptyNeighborhood = False
            
            b += 1
            genX.append(newX)

            if b == br:
                break
                        
    return genX, budgetLeft, i1

# Function to Generate New Solutions (Hybrid Neighborhood)
def getNewSolutions_Hybrid(surSols, br, i1, r, R, budgetLeft, d, E, X, seed_I1):
    
    # Solution Search Steps
    numExplore1 = int(round(R*1/10, 0))                                 
    numExplore2 = int(round(R*5/10, 0))                                 
    numExploit1 = int(round(R*2/10, 0))                                 
    numExploit2 = int(R - numExplore1 - numExplore2 - numExploit1)      
    
    startExplore1 = 1                                                 
    startExplore2 = startExplore1 + numExplore1                     
    startExploit1 = startExplore2 + numExplore2                        
    startExploit2 = startExploit1 + numExploit1                    
    
    b = 0
    genX = []
    
    while b < br:
        for idx in surSols:
            
            ## Get Neighborhood Search Location
            # Start Explore 1
            if r < startExplore2:
                m = math.ceil(d/1/r)
            # Start Explore 2
            elif r < startExploit1:
                m = math.ceil(d/1.25/r)
            # Start Exploit 1
            elif r < startExploit2:
                m = math.ceil(d/5/r)
            # Start Exploit 2
            else:
                m = math.ceil(d/10/r)

            # Begin Search
            emptyNeighborhood = True
            tempE = [X[e] for e in E]
            #Check if the m-neighborhood of Xi is empty
            while emptyNeighborhood == True:
                
                #Uniformly select one new solution from the m-neighborhood of Xi
                uniqueSol = False
                #print("d = ", d, " | m = ", m)
                HmSize = int((math.factorial(d))/((math.factorial(m))*(math.factorial(d-m))))
                HmCounter = 0
                while uniqueSol == False:
                    i1 += 1
                    HmCounter += 1
                    
                    random.seed(seed_I1[i1])
                    tempSel = list(random.sample(range(d),m))            

                    Xi = np.array(copy.deepcopy(X[idx]))
                    Xi[tempSel] = 1 - Xi[tempSel]
               
                    if list(Xi) not in X:
                        if list(Xi) not in genX:
                            if list(Xi) not in tempE:
                                newX = list(Xi)
                                uniqueSol = True
                                break
                    
                    if HmCounter >= HmSize:
                        break
                                                
                if uniqueSol == False:
                    m-= 1
                    if m == 0:
                        m = d
                else:
                    emptyNeighborhood = False
            
            b += 1
            genX.append(newX)
            
            if b == br:
                break
                        
    return genX, budgetLeft, i1


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