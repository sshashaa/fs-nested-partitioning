# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:45:37 2024

@author: ephouser
"""

from PythonModules import *
from RS_SolutionSearch import *
from EstimatingPerformance import *
from GeneratingDatasets import *

#########################################################################################
#########################################################################################

########################
# Rapid Screening Driver
def prepareScreening(inputsExpSettings, inputsData, inputsParameters):
    
    ###############
    # Unpack Inputs
    computer, INDIV, dataSize, EXP_I, EXP_J, EXP_J_STR, EXP_K, expL_Screen = inputsExpSettings
    trueSol, dataX_All, dataX_MacJ, dataY_MacJ, dataTrainValidMic = inputsData
    RS_Q_Percent, RS_P_Percent, RS_ITR, iSR, seedTrainTestMic = inputsParameters 
    
    dataX_Train_MacJ, dataX_Test_MacJ = dataX_MacJ
    dataY_Train_MacJ, dataY_Test_MacJ = dataY_MacJ
    repScreening, screenTime, screenTimeMin = [], [], []
    
    dataInstance = dataSize+EXP_J_STR
    
    ######################################################################################################################################################
    
    ##############################
    # USE ORIGINAL RAPID SCREENING
    if EXP_K == 'ORS':
        
        #################################
        # FOR EACH SOLUTION SEARCH METHOD
        for EXP_L in range(len(expL_Screen)):

            #################
            # START SCREENING
            screenStartTime = getTimeNow()
            searchType = expL_Screen[EXP_L]     # Set Search Method ('Nearest', 'Shrinking', or 'Hybrid')      
            
            ##########################
            # INITIATE RAPID SCREENING
            tempInputsExpSettings = [computer, dataInstance, EXP_J, EXP_K]
            tempInputsParameters = [searchType, RS_Q_Percent, RS_P_Percent, RS_ITR, iSR, seedTrainTestMic]
            tempInputsData = [dataX_Train_MacJ, dataY_Train_MacJ, dataTrainValidMic]                                 
            keptFeats, npOrder, iSR, dataTrainValidMic, modelName = getOrder(tempInputsExpSettings, tempInputsParameters, tempInputsData)     
            
            #########################
            # Track Replications Used
            repScreening.append(iSR)
            print(f"{repScreening[EXP_L]} Replications Used in Screening")
            
            ##################
            # GET CURRENT TIME
            currentTime = getTimeNow()   
            
            #######################################################
            # CALCULATE RUN TIME (CONSIDERING POSSIBLE DATE CHANGE)
            if currentTime < screenStartTime:
                currentTime += timedelta(days=1)    
        
            ############################
            # STORE TOTAL SCREENING TIME
            screenTime.append(currentTime-screenStartTime)
            screenTimeMin.append(round(screenTime[EXP_L].total_seconds() / 60, 4))
            
            #########################
            # COMPILE TIME MILESTONES            
            timeSummary = [modelName, dataInstance, screenTimeMin[EXP_L], 0, screenTimeMin[EXP_L]]
            timeSummary = pd.DataFrame(data = [timeSummary], columns = ["Model" , "Data Instance", "Screen Time", "Partition Time", "Total Time"])
            
            #######################################
            # FILTER OUT FEATURES THAT WERE DROPPED
            featSpace = keptFeats
            for dataDict in dataTrainValidMic.values():
                for k,v in dataDict.items():
                    if 'x' in k.lower():
                        dataDict[k] = v.loc[:,featSpace]
            
            dataX = copy.deepcopy(dataX_All)            
            dataX = dataX.loc[:,featSpace]
            dataX_Train_MacJ = dataX_Train_MacJ.loc[:,featSpace]
            dataX_Test_MacJ = dataX_Train_MacJ.loc[:,featSpace]
            
            dataX_Filtered = [dataX, dataX_Train_MacJ, dataX_Test_MacJ]
            
            ########################
            # COMPILE FINAL FEATURES
            finalFeats = keptFeats
            
            ####################################
            ####################################
            ###  COMPILE AND EXPORT RESULTS  ###
            ####################################
            ####################################
            
            ###########################
            # COMPILE SCREENING RESULTS
            colNames = ["Model", "Data Instance", "Solution", "Time"]
            tempScreenResults = [modelName, dataInstance, finalFeats, screenTimeMin[EXP_L]]
            screenResultsDF = pd.DataFrame(data = [tempScreenResults], columns = colNames)
            
            countTrue = 0
            tempRS_SOL = screenResultsDF['Solution'].iloc[0]
            for xI in range(len(tempRS_SOL)):
                if 'x' in tempRS_SOL[xI]:
                    countTrue += 1
                    
            screenResultsDF['Percent Correct'] = round(countTrue / len(trueSol),3)
            screenResultsDF['Total Feats'] = len(tempRS_SOL)
            screenResultsDF['Replications'] = iSR

            #############################
            ###  EXPORT SCREENING RESULTS
            # Prepare File Path
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
                resultsSummary = pd.concat([resultsSummary,screenResultsDF], axis = 0, ignore_index=True)
                
            elif ((resultsSummary['Model'] == screenResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == screenResultsDF.iloc[0,1])).any():
                
                # Replace the existing row
                resultsSummary.loc[(resultsSummary['Model'] == screenResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == screenResultsDF.iloc[0,1])] = screenResultsDF.values
                
            else:
                resultsSummary = pd.concat([resultsSummary,screenResultsDF], axis = 0, ignore_index=True)
                
            # Export Results
            resultsSummary.to_excel(resultsSummary_Path, index = False)
                
            #######################
            ### EXPORT TIME RESULTS    
            
            # Prepare File Path
            if computer == 0:
                resultsTime_Path = "/home/ephouser/RSP/Results/RSP_TimeBreakdown.xlsx"
            else:
                resultsTime_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\RSP_TimeBreakdown.xlsx"

            # Import Time Results
            if os.path.isfile(resultsTime_Path):
                resultsTime = pd.read_excel(resultsTime_Path, index_col=None)
            else:
                resultsTime = pd.DataFrame(data = None, columns = ["Model", "Data Instance", "Screen Time", "Partition Time", "Total Time"])
            
            
            # Add/Update Results DF
            if len(resultsTime) == 0:
                # Add row to the end of the dataframe
                resultsTime = pd.concat([resultsTime,timeSummary], axis = 0, ignore_index=True)
                
            elif ((resultsTime['Model'] == timeSummary.iloc[0,0]) & (resultsTime['Data Instance'] == timeSummary.iloc[0,1])).any():
                # Replace the existing row
                resultsTime.loc[(resultsTime['Model'] == timeSummary.iloc[0,0]) & (resultsTime['Data Instance'] == timeSummary.iloc[0,1])] = timeSummary.values
                
            else:
                resultsTime = pd.concat([resultsTime,timeSummary], axis = 0, ignore_index=True)
                
            # Export Results
            resultsTime.to_excel(resultsTime_Path, index = False)  
   
    ######################################################################################################################################################

    ##################################
    # USE ORIGINAL NESTED PARTITIONING
    if EXP_K == 'NP':
        
        for EXP_L in range(len(expL_Screen)):
            
            #################
            # START SCREENING
            searchType = expL_Screen[EXP_L]     # Set Search Method ('Nearest', 'Shrinking', or 'Hybrid')      

            ##########################
            # INITIATE RAPID SCREENING
            tempInputsExpSettings = [computer, dataSize, EXP_J, EXP_K]
            tempInputsParameters = [searchType, RS_Q_Percent, RS_P_Percent, RS_ITR, iSR, seedTrainTestMic]
            tempInputsData = [dataX_Train_MacJ, dataY_Train_MacJ, dataTrainValidMic]                                 
            keptFeats, npOrder, iSR, dataTrainValidMic, modelName = getOrder(tempInputsExpSettings, tempInputsParameters, tempInputsData)  
    
            keptFeats = pd.DataFrame(data = None, columns = ["Variable", "Decision"])

            # Track Replications Used
            repScreening.append(0)
            print(f"{repScreening[EXP_L]} Replications Used in Screening")
            
            tempScreenTime = timedelta(0)
            screenTime.append(tempScreenTime)
            screenTimeMin.append(0) 
            
            dataX = copy.deepcopy(dataX_All)   
            dataX_Filtered = [dataX, dataX_Train_MacJ, dataX_Test_MacJ]
    
    ######################################################################################################################################################

    #########################
    # USE NEW RAPID SCREENING
    if EXP_K == 'RSP':
        
        #################################
        # FOR EACH SOLUTION SEARCH METHOD
        for EXP_L in range(len(expL_Screen)):

            #################
            # START SCREENING
            screenStartTime = getTimeNow()
            searchType = expL_Screen[EXP_L]     # Set Search Method ('Nearest', 'Shrinking', or 'Hybrid')      

            ##########################
            # INITIATE RAPID SCREENING
            tempInputsExpSettings = [computer, dataSize, EXP_J, EXP_K]
            tempInputsParameters = [searchType, RS_Q_Percent, RS_P_Percent, RS_ITR, iSR, seedTrainTestMic]
            tempInputsData = [dataX_Train_MacJ, dataY_Train_MacJ, dataTrainValidMic]                                 
            keptFeats, npOrder, iSR, dataTrainValidMic, modelName = getOrder(tempInputsExpSettings, tempInputsParameters, tempInputsData)     
            
            #########################
            # Track Replications Used
            repScreening.append(iSR)
            print(f"{repScreening[EXP_L]} Replications Used in Screening")
            
            ##################
            # GET CURRENT TIME
            currentTime = getTimeNow()   
            
            #######################################################
            # CALCULATE RUN TIME (CONSIDERING POSSIBLE DATE CHANGE)
            if currentTime < screenStartTime:
                currentTime += timedelta(days=1)    
        
            ############################
            # STORE TOTAL SCREENING TIME
            screenTime.append(currentTime-screenStartTime)
            screenTimeMin.append(round(screenTime[EXP_L].total_seconds() / 60, 4))
            
            #########################
            # COMPILE TIME MILESTONES            
            timeSummary = [modelName, dataInstance, screenTimeMin[EXP_L], 0, screenTimeMin[EXP_L]]
            timeSummary = pd.DataFrame(data = [timeSummary], columns = ["Model" , "Data Instance", "Screen Time", "Partition Time", "Total Time"])
            
            #######################################
            # FILTER OUT FEATURES THAT WERE DROPPED
            featSpace = pd.concat([keptFeats,npOrder], axis = 0, ignore_index = True)
            for dataDict in dataTrainValidMic.values():
                for k,v in dataDict.items():
                    if 'x' in k.lower():
                        dataDict[k] = v.loc[:,featSpace['Variable']]
            
            dataX = copy.deepcopy(dataX_All)            
            dataX = dataX.loc[:,featSpace['Variable']]
            dataX_Train_MacJ = dataX_Train_MacJ.loc[:,featSpace['Variable']]
            dataX_Test_MacJ = dataX_Train_MacJ.loc[:,featSpace['Variable']]
            
            dataX_Filtered = [dataX, dataX_Train_MacJ, dataX_Test_MacJ]
            
            ####################################
            ####################################
            ###  COMPILE AND EXPORT RESULTS  ###
            ####################################
            ####################################
            
            ###########################
            # COMPILE SCREENING RESULTS
            colNames = ["Model", "Data Instance", "Solution", "Time"]
            tempScreenResults = [modelName, dataInstance, list(keptFeats['Variable'])+list(npOrder['Variable']), screenTimeMin[EXP_L]]
            screenResultsDF = pd.DataFrame(data = [tempScreenResults], columns = colNames)
            
            countTrue = 0
            tempRS_SOL = screenResultsDF['Solution'].iloc[0]
            for xI in range(len(tempRS_SOL)):
                if 'x' in tempRS_SOL[xI]:
                    countTrue += 1
                    
            screenResultsDF['Percent Correct'] = round(countTrue / len(trueSol),3)
            screenResultsDF['Total Feats'] = len(tempRS_SOL)
            screenResultsDF['Replications'] = iSR

            #############################
            ###  EXPORT SCREENING RESULTS
            # Prepare File Path
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
                resultsSummary = pd.concat([resultsSummary,screenResultsDF], axis = 0, ignore_index=True)
                
            elif ((resultsSummary['Model'] == screenResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == screenResultsDF.iloc[0,1])).any():
                
                # Replace the existing row
                resultsSummary.loc[(resultsSummary['Model'] == screenResultsDF.iloc[0,0]) & (resultsSummary["Data Instance"] == screenResultsDF.iloc[0,1])] = screenResultsDF.values
                
            else:
                resultsSummary = pd.concat([resultsSummary,screenResultsDF], axis = 0, ignore_index=True)
                
            # Export Results
            resultsSummary.to_excel(resultsSummary_Path, index = False)
                
            #######################
            ### EXPORT TIME RESULTS    
            
            # Prepare File Path
            if computer == 0:
                resultsTime_Path = "/home/ephouser/RSP/Results/RSP_TimeBreakdown.xlsx"
            else:
                resultsTime_Path = "G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\RSP_TimeBreakdown.xlsx"

            # Import Time Results
            if os.path.isfile(resultsTime_Path):
                resultsTime = pd.read_excel(resultsTime_Path, index_col=None)
            else:
                resultsTime = pd.DataFrame(data = None, columns = ["Model", "Data Instance", "Screen Time", "Partition Time", "Total Time"])
            
            
            # Add/Update Results DF
            if len(resultsTime) == 0:
                # Add row to the end of the dataframe
                resultsTime = pd.concat([resultsTime,timeSummary], axis = 0, ignore_index=True)
                
            elif ((resultsTime['Model'] == timeSummary.iloc[0,0]) & (resultsTime['Data Instance'] == timeSummary.iloc[0,1])).any():
                # Replace the existing row
                resultsTime.loc[(resultsTime['Model'] == timeSummary.iloc[0,0]) & (resultsTime['Data Instance'] == timeSummary.iloc[0,1])] = timeSummary.values
                
            else:
                resultsTime = pd.concat([resultsTime,timeSummary], axis = 0, ignore_index=True)
                
            # Export Results
            resultsTime.to_excel(resultsTime_Path, index = False)  

    
    inputsExpSettings = [computer, INDIV, dataSize, EXP_I, EXP_J, EXP_J_STR, EXP_K, EXP_L, expL_Screen]
    savedResults = [inputsExpSettings, repScreening, screenTime, screenTimeMin, keptFeats, npOrder, iSR, dataTrainValidMic, dataX_All, dataX_Filtered, modelName]

    return savedResults
            
#########################################################################################
#########################################################################################

##################################################
# RAPID SCREENING TO DEFINE PARTITIONING STRUCTURE
def getOrder(tempInputsExpSettings, tempInputsParameters, tempInputsData):

    ###############
    # UNPACK INPUTS
    computer, dataSize, EXP_J, EXP_K = tempInputsExpSettings
    searchType, RS_Q_Percent, RS_P_Percent, RS_ITR, iSR, seedTrainTestMic = tempInputsParameters
    dataX_Train_MacJ, dataY_Train_MacJ, dataTrainValidMic = tempInputsData
    
    #####################
    # RUN RAPID SCREENING  
    varSummary, iSR, dataTrainValidMic, positionHistory, modelName = screenData(dataX_Train_MacJ, dataY_Train_MacJ, dataTrainValidMic, 
                                                                                searchType, iSR, seedTrainTestMic, dataSize, EXP_J, EXP_K, computer, RS_ITR)    
    ####################################
    # IMPLEMENT ORIGINAL RAPID SCREENING
    if EXP_K == 'ORS':

        keptFeats = varSummary
        partOrder = varSummary      
        
    ####################################
    # IMPLEMENT NEW RAPID SCREENING
    if EXP_K == 'RSP':
        
        #############
        # THRESHOLDS: 
        incP = int(math.ceil(len(dataX_Train_MacJ.columns)*RS_P_Percent))       # TOP P% AUTOMATICALLY CHOSEN
        excQ = int(math.ceil(len(varSummary)*(1-RS_Q_Percent)))                   # BOTTOM Q% SCREENED OUT
        # varQ = np.quantile(varSummary["Variance"].values, RS_Q_Percent)         # KEEP HIGH VAR FEATS IN BOTTOMG Q% (WHAT IS "HIGH VAR"?)   
        
        ####################################
        # MAKE DECISIONS BASED ON THRESHOLDS
        tempDec = []
        for varI in range(1,len(varSummary)+1):
            if varI <= incP:                                        # IF FEAT IN TOP P% (KEEP)
                tempDec.append(1)
            # elif varSummary["Variance"].loc[varI] >= varQ:
            #     tempDec.append(0) # Change to -1 if using this
            elif varI > excQ:                                       # IF FEAT IN BOTTOM Q% (DROP)
                tempDec.append(0)
            else:                                                   # ALL FEATS IN BETWEEN (TBD)
                tempDec.append(-1)
        
        #################
        # STORE DECISIONS
        varSummary["Decision"] = tempDec
        
        ################################
        # CREATE KEPT AND TBD FEATS LIST
        keptFeats = varSummary.loc[varSummary["Decision"] == 1].reset_index(drop=True)      # FEATS KEPT
        partOrder = varSummary.loc[varSummary["Decision"] == -1].reset_index(drop=True)     # FEATS TO BE DECIDED ON
        
        tempVarSummaryKeep = varSummary[["Variable", "Decision"]]                           # COMPILE SCREENING RESULTS

    ##########################
    # RAPID SCREENING NOT USED
    if EXP_K == 'NP':
        keptFeats = list(dataX_Train_MacJ.columns)
        partOrder = pd.DataFrame(data = np.array([dataX_Train_MacJ.columns, [-1]*len(dataX_Train_MacJ.columns)]).T, columns= ['Variable', 'Decision'])
    
    return keptFeats, partOrder, iSR, dataTrainValidMic, modelName

#########################################################################################
#########################################################################################

###########################
# Robust Screening Function
def screenData(dataX_Train_MacJ, dataY_Train_MacJ, dataTrainValidMic, searchType, iSR, seedTrainTest, dataSize, EXP_J, EXP_K, computer, RS_ITR):
        
    ##########################################
    # Get Time for Tracking Iteration Duration
    startTimeTracking = getTimeNow()
    
    ########################################
    ######  Parameter Initialization  ######
    ########################################
    
    # User-Specified
    d=len(dataX_Train_MacJ.columns)                     # Number of Features
    M = math.ceil(d*math.log10(d)/10)*10                # User-specified integer constant 
    R = RS_ITR                                          # R: Number of screening Iterations
    b=[50]                                              # b(r): Number of new solutions in the rth iteration
    n0 = 10                                             # Initial Number of Samples
    G = 1.2                                             # Replication Growth Factor
    deltaS = 0.05                                       # Indifference Zone Parameter
    
    if EXP_K == "RSP":
        n0 = 1
        G = 1.1
    else:
        n0 = 10
        G = 1.2
    
    # Naming Conventions
    if EXP_K == "ORS":    
        modelName = f"ORS{R}-{searchType}"
    elif EXP_K == "RSP":
        modelName = f"RS{R}-{searchType}_n{n0}"
        
        
    EXP_J_STR = "0" + str(EXP_J+1) if (EXP_J+1) < 10 else str(EXP_J+1)
    dataInstance = dataSize+EXP_J_STR
    
    modelType = 0                                   # 0 = Linear Regression, 1 = Logistic Regression
    trainSize = 0.8                                 # Size of Training Set
    testSize = round(1-trainSize,2)                 # Size of Testing Set
                                          
    alpha1 = 0.025                                  # Significance Levels
    alpha2 = 0.025
    alpha = alpha1 + alpha2
    alphaPrime = 1-(1-alpha1)**(1/R)

    ITilda = [[]]                                                   # I_Tilda(r): Indices of the b(r) solutions
    IPrime=[[]]                                                     # Set of Surviving Solutions through iteration r-1
    I = []                                                          # Union of surviving solutions and b(r) new solutions at the beginning of the rth iteration
    B= []                                                           # Cumulative number of solutions visited up to iteration r
    n = []                                                          # Number of replications allocated to each solution at iteration r (Round UP: n0*G**r)
    r = 0                                                           # Current iteration
    Ni= pd.DataFrame(np.zeros((R+2, 5*M*R)))                        # Total number of observations on solution Xi up to iteration r for ever i in I(r)
    Ybari = []                                                      # Sample average
    Wil = []                                                        # Screening Threshold when comparing solutions Xi and Xl at iteration r
    X = []                                                          # Set of all solutions
    
    Yij  = pd.DataFrame(data = None, columns = list(range(5*M*R)))  # Set of all Yij
    for itr in range(R+2):
        tempDF = pd.DataFrame(data = [[[]]*5*M*R])
        Yij = pd.concat([Yij, tempDF], axis = 0, ignore_index=True)
        
    Y = []                                                          # Set of all Ybars
    E = []                                                          # Set of Excluded Solutions
    i1 = 0                                                          # Solution Generator Seed Counter
    countImprovement = 0                                            # Counter for Tracking Iteration Improvement
   
    # Manage Computational Budget
    budget = 100000  # Sampling Budget
    budgetLeft = True
    
    random.seed(321)
    seed_NewSol = random.sample(range(1000000),1000000) 

    random.seed(123)
    seedsM = random.sample(range(1000000),1000000)      
    
    random.seed(0)
    seed_I0 = random.sample(range(100000), 100000)
    
    # Seeds
    def generate_seed_list(seed):
        random.seed(seed)
        return list(random.sample(range(1000000), 1000000))
    
    nItrs = 10
    seed_I1_parallel = Parallel(n_jobs=-1)(delayed(generate_seed_list)(seed) for seed in range(nItrs))
    seed_I1 = [seed for sublist in seed_I1_parallel for seed in sublist]
    
    s = 0   # Helps track seed index for CRN
    
    #####################################
    ########  Create Dataframes  ########
    #####################################
    
    positionHistory = pd.DataFrame(data = None, columns = ["Variable"])                       # DataFrame Tracking Feat Rank Over Time
    myR = list(range(5,int(math.ceil(R / 5.0)) * 5+5,5))

    sideExpRanksFrequency = pd.DataFrame(data=None)
    sideExpRanksAverage = pd.DataFrame(data=None)
    sideExpStats = pd.DataFrame(data = None, columns = ['R', 'Time', 'Perf Mean', 'Perf Var', 'Perf Improvement'])
    
    convPlotMean, convPlotLB, convPlotUB = [], [], []
    
    print("#############################################")
    print("############## BEGIN SCREENING ##############")
    print("#############################################")
    print("\n")


    for r in range(0,R+1):
        
        print("#######################################")
        print(f"        START ITERATION {r}/{R+1}")
        print(f"             RS-{searchType} {EXP_J_STR}")
        print("#######################################")
        
        
        newSols = []                                                # Reset list of new solutions
        tempITilda = []                                             # Reset list of indices for new solutions
        n.append(int(min(max(math.ceil(n0*G**r),2), 1000)))                # Number of replications allocated to each solution at iteration r
        
        
        #####################################            
        #####################################
        ######    FIND NEW SOLUTIONS    #####
        #####################################
        #####################################            
        
        #Step 1 of Algorithm
        if r == 0:
            #Obtain Initial Solutions
            newSols, budgetLeft = getInitialSolutions(d, b[0], budgetLeft, seed_I0)
            X = X + newSols
            
            #Update ITilda
            tempITilda = [i for i in range(0,b[0])]
            ITilda.append(tempITilda)
            
            #Update the cumulative num of solutions
            B.append(b[0])
            
            #Union of b(r) new solutions and IPrime surviving solutions
            I.append(tempITilda)
            
            #Update num of replications needed for each new Xi
            for i in tempITilda:
                Ni.iloc[r,i] = int(n[r])
        
        else:
            #Generate New Solution
            # b.append(M)
            b.append(max(M-len(IPrime[r]),0))
            #s += b[r]           #Set new seeds for next iteration
            
            print("")
            print("Solution Search: BEGAN!")
            print("")
            
            #Obtain New Solutions
            if searchType == 'Nearest':
                newSols, budgetLeft, i1 = getNewSolutions_Nearest(IPrime[r], b[r], i1, r, budgetLeft, d, E, X, seed_I1)
            elif searchType == 'Shrinking':
                newSols, budgetLeft, i1 = getNewSolutions_Shrinking(IPrime[r], b[r], i1, r, budgetLeft, d, E, X, seed_I1)
            elif searchType == 'Hybrid':
                newSols, budgetLeft, i1 = getNewSolutions_Hybrid(IPrime[r], b[r], i1, r, R, budgetLeft, d, E, X, seed_I1)
 
            print("Solution Search: COMPLETE!")
            print("")
    
            X = X + newSols
            
            #Update I Tilda
            tempITilda = [B[r-1]+i for i in range(0,len(newSols))]
            ITilda.append(tempITilda)
            
            #Update cumulative num of solutions
            B.append(B[r-1]+b[r])
            
            #Union of b(r) new solutions and IPrime surviving solutions            
            I.append(IPrime[r]+tempITilda)
            
            #Update num of replications needed for each new Xi
            for i in tempITilda:
                
                if len(Ni.columns) < tempITilda[-1]:
                    tempNi = pd.DataFrame(np.zeros((R+2, abs(tempITilda[-1] - len(Ni.columns)))))
                    Ni = pd.concat([Ni, tempNi], axis = 1)
                    
                Ni.iloc[r-1, i] = 0

            #Update num of replications needed for each new Xi
            for i in I[r]:
                Ni.iloc[r,i] = int(Ni.iloc[r-1, i] + n[r])



        ############################################            
        ############################################
        ######    FIND SOLUTION PERFORMANCE    #####
        ############################################
        ############################################     

        #Step 2 of Algorithm (Approach B)
        Ybari = []
        for i in I[r]:

            # Get Solution
            tempNewSol = np.array(X[i])
            
            if sum(tempNewSol) == 0:            
                Yij.iloc[r,i] = np.array([-1*99999 for iVal in range(n[-1])])                    
                Ybari.append(-1*99999)
                
            else:
                if modelType == 0:
                    
                    if Ni.iloc[r,i] > len(dataTrainValidMic):
                        dataTrainValidMic = getMoreTrainTestData(dataX_Train_MacJ, dataY_Train_MacJ,testSize, 
                                                                  int(Ni.iloc[r,i]) - len(dataTrainValidMic), seedTrainTest, dataTrainValidMic)                    
                    iSR += n[-1]
                    
                    
                    repStart = int(Ni.iloc[r,i] - n[-1])
                    repEnd = int(Ni.iloc[r,i])
                    tempYij = []                    
                    for j in range(repStart, repEnd):
                        temp_trainX = dataTrainValidMic[j]['trainX']
                        temp_testX = dataTrainValidMic[j]['testX']
                        temp_trainY = dataTrainValidMic[j]['trainY']
                        temp_testY  = dataTrainValidMic[j]['testY']
                        
                        tempMAE, tempMSE, tempRMSE = fitLinearReg(temp_trainX.loc[:,tempNewSol==1],
                                                                  temp_testX.loc[:,tempNewSol==1],
                                                                  temp_trainY,
                                                                  temp_testY)
                        tempYij.append(-1*tempRMSE)
                    
                    if i <= len(Yij.columns):
                        Yij.iloc[r,i] = list(np.array(tempYij))                    
                    else:
                        tempYijDF = pd.DataFrame(np.zeros((R+2, i - len(Yij.columns))))
                        Yij = pd.concat([Yij, tempYijDF], axis = 1)
                        Yij.iloc[r,i] = list(np.array(tempYij))    
                    
                    Ybari.append(round((1/n[-1])*sum(Yij.iloc[r,i]),5))

                        
                # elif modelType == 1:
                #     #Compute Sample Average Ybari
                #     tempYij = []
                #     for j in range(numReps):
                #         iSR += 1
                        
                #         # Build Train and Test Datasets
                #         trainX, testX, trainY, testY = train_test_split(dataX.loc[:,tempNewSol==1], dataY, test_size = 0.2, random_state=seedsM[iSR])
                        
                #         #################################################
                #         ## ERROR HERE FOR MULTINOMIAL LOG REG ROC CURVE##
                #         #################################################
                        
                #         f1_score, f2_score, f3_score, f4_score, precision, recall = fitLogReg(trainX, testX, trainY, testY)
                #         tempData = [f1_score, f2_score, f3_score, f4_score, precision, recall]
                #         tempYij.append(f3_score)
            
                #     Yij.append(np.array(tempYij))
                #     Ybari.append(round((1/numReps)*sum(tempYij),3))                


        ####################################################################################################################################################
        ### CODE FOR PARALLEL PROCESSING

        # if r==0:
        #     numPrevReps = 0 
        # else:
        #     numPrevReps = int(Ni.iloc[r-1, i])

        # numReps = int(Ni.iloc[r, i])
        # numNewReps = int(Ni.iloc[r, i] - numPrevReps)
        
        # iSR += numNewReps
        
        # repInfo = [numPrevReps, numReps, numNewReps]

        # tempYbari = []
        # if i in Yij.loc[:,"Index"]:
        #     tempYij = list(Yij.iloc[i,1])
        #     print(tempYij)
        # else:
        #     tempYij = []
        
        # Using joblib to parallelize the loop and collecting the results
        # results = Parallel(n_jobs= -1, verbose = 10)(
        #     delayed(process_iteration)(i, r, repInfo, Ni, X, tempYij, tempYbari, modelType, dataTrainValidMic) for i in I[r]
        # )
        
        # Collecting modified Yij and Ybari values after parallel execution
        # for result in results:
        #     i, Yij_result, Ybari_result = result
                        
        #     if i in Yij.loc[:,"Index"]:
        #         Yij.iloc[:,1][i] = np.array(list(Yij.iloc[i,1]) + list(np.array(Yij_result)))
        #     else:
        #         Yij.loc[len(Yij)] = [i, np.array(Yij_result)]
        #     Ybari.extend(Ybari_result)

        # print(Yij)
        # print(Ybari)

        ####################################################################################################################################################
        
        
        #################################################            
        #################################################
        ######    IDENTIFY & SAVE BEST SOLUTIONS    #####
        #################################################
        #################################################  

        Y.append(Ybari)
                   
        solSummary = []
        for i in range(len(I[r])):
            temp = []
            temp.append(Ybari[i])
            for j in X[I[r][i]]:
                temp.append(j)
            solSummary.append(temp)
        
        checkColNames = dataX_Train_MacJ.columns
        checkColNames = checkColNames.insert(0,"Ybar")
        solSummaryDF = pd.DataFrame(data=solSummary, columns=checkColNames)                
        
        
        ####################################################################
        ####################################################################
        ######    Compute Screening Threshold Wil for all i,l, i!=l    #####
        ####################################################################
        ####################################################################
        Wil = np.empty((len(I[r]),len(I[r])))
        
        for i in range(len(I[r])):
            for l in range(len(I[r])):

                q = 1-(alphaPrime/(len(I[r])-1))
                dfil = n[-1] - 1  
                til = scipy.stats.t.ppf(q = q, df = dfil)
                Sil = math.sqrt((1/dfil)*sum((np.array(Yij.iloc[r, I[r][i]]) - np.array(Yij.iloc[r, I[r][l]]) - (Ybari[i] - Ybari[l]))**2))
                    
                if i == l:
                    Wil[i,l] = 0.000
                else:
                    Wil[i,l] = round((til)*(Sil)/(math.sqrt(n[-1])), 6)
        
        
        ################################################################
        ################################################################
        #####    Check To See Which Solutions Should Be Removed    #####
        ################################################################
        ################################################################
        YbarDiff = np.empty((len(I[r]),len(I[r])))
        WithinThreshold = np.empty((len(I[r]),len(I[r])))

        for i in range(len(I[r])):
            for l in range(len(I[r])):
                YbarDiff[i,l] = Ybari[i] - Ybari[l]
                
                if i == l:
                    WithinThreshold[i,l] = True
                elif YbarDiff[i,l] <= -1*(Wil[i,l]):    # DO WE USE deltaS HERE???
                    WithinThreshold[i,l] = False
                else:
                    WithinThreshold[i,l] = True


        #####################################################
        #####################################################
        #####    Decide To Keep or Discard Solutions    #####
        #####################################################
        #####################################################

        tempIPrime = []
        Ybari_SurvSol = [] 
        for i in range(len(I[r])):
            if sum(WithinThreshold[i,:]) == len(I[r]):
                tempIPrime.append(I[r][i])
                Ybari_SurvSol.append(Ybari[i])
            else:
                if I[r][i] not in E:
                    E.append(I[r][i])
                
        IPrime.append(tempIPrime)
        
        
        ############################################################
        ### Compile Surviving Solutions & Corresponding Performances
        solSummaryFinal = []
        for i in range(len(I[r])):
           temp = []
           
           if I[r][i] in IPrime[r+1]:
               temp.append(Y[-1][i])
               
               for j in IPrime[r+1]:
                   if j == I[r][i]:
                                          
                       for k in X[j]:
                           temp.append(k)
                       solSummaryFinal.append(temp)
 
        
        checkColNames = dataX_Train_MacJ.columns
        checkColNames = checkColNames.insert(0,"Ybar")
        solSummaryFinalDF = pd.DataFrame(data=solSummaryFinal, columns=checkColNames)
        
        if EXP_K == 'RSP':        
            if computer == 0:
                solSummaryFinalDF.to_excel(f"/home/ephouser/RSP/Results/SideExp/SideExpSurvSol_and_Perf_{searchType}_{dataInstance}.xlsx", index = False)  
            else:
                solSummaryFinalDF.to_excel(f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\SideExpSurvSol_and_Perf_{searchType}_{dataInstance}.xlsx", index = False)  
        
        ###########################################################################################################################
        
        if EXP_K == "RSP":
        
            ########################################
            ########################################
            ### SIDE EXP: PREP CONVERGENCE PLOTS ###
            ########################################
            ########################################
            
            tempSurvSolMean = round(-1*solSummaryFinalDF['Ybar'].mean(),6)
            tempSurvSolVar = round(solSummaryFinalDF['Ybar'].var(),6)        
            tempSurvSolLB = tempSurvSolMean - scipy.stats.t.ppf(0.975, n[-1])*math.sqrt(tempSurvSolVar/n[-1])
            tempSurvSolUB = tempSurvSolMean + scipy.stats.t.ppf(0.975, n[-1])*math.sqrt(tempSurvSolVar/n[-1])
            
            convPlotMean.append(tempSurvSolMean)
            convPlotLB.append(tempSurvSolLB)
            convPlotUB.append(tempSurvSolUB)
                         
            ##########################
            # COMPILE CONVERGENCE DATA
            
            tempPlotSummaryDF = pd.DataFrame(data = [[dataInstance, r, tempSurvSolMean, tempSurvSolLB, tempSurvSolUB]], columns = ["Data Instance", "Itr", "Mean", "LB", "UB"])
            
            if computer == 0:
                plotSummary_Path = f"/home/ephouser/RSP/Results/SideExp/AllConvPlotData_{dataSize}-{searchType}.xlsx"
            else:
                plotSummary_Path = f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\AllConvPlotData_{dataSize}-{searchType}.xlsx"
    
            # Import Results (If Existing)
            if os.path.isfile(plotSummary_Path):
                plotSummary = pd.read_excel(plotSummary_Path, index_col=None)
            else:
                plotSummary = pd.DataFrame(data = None, columns = ["Data Instance", "Itr", "Mean", "LB", "UB"])
                            
            # Add/Update Results DF
            if len(plotSummary) == 0:
                # Add row to the end of the dataframe
                plotSummary = pd.concat([plotSummary,tempPlotSummaryDF], axis = 0, ignore_index=True)
                
            elif ((plotSummary["Data Instance"] == tempPlotSummaryDF.iloc[0,0]) & (plotSummary["Itr"] == tempPlotSummaryDF.iloc[0,1])).any():
                # Replace the existing row
                plotSummary.loc[(plotSummary["Data Instance"] == tempPlotSummaryDF.iloc[0,0]) & (plotSummary["Itr"] == tempPlotSummaryDF.iloc[0,1])] = tempPlotSummaryDF.values
                
            else:
                plotSummary = pd.concat([plotSummary,tempPlotSummaryDF], axis = 0, ignore_index=True)
                
            # Export Results
            plotSummary.to_excel(plotSummary_Path, index = False)

            ###########################################################################################################################
    
            #############################
            # Track Iteration Improvement
            if r < 1:
                tempSurvSolImprov = 0
                percentChange = 0
            else:
                tempSurvSolImprov = tempSurvSolMeanPrevious - tempSurvSolMean
                
                percentChange = (tempSurvSolMeanPrevious - tempSurvSolMean) / tempSurvSolMeanPrevious
                
                if abs(percentChange) <= 0.03:
                    countImprovement += 1
                else:
                    countImprovement = 0
            
            ##################
            # GET CURRENT TIME
            currentTimeTracking = getTimeNow()
            
            if currentTimeTracking < startTimeTracking:
                # Handle the date change by adding a day to the end_time
                currentTimeTracking += timedelta(days=1)
    
            ##########################
            # STORE ITERATION DURATION            
            tempTime = currentTimeTracking-startTimeTracking
            tempTimeMin = round(tempTime.total_seconds() / 60, 4)
        
            ###########################################################################################################################
                
            ##########################################
            ##########################################
            ### SIDE EXP: Compile and Export Stats ###
            ##########################################
            ##########################################
            tempSideExpStats = pd.DataFrame(data = [[r, tempTimeMin, tempSurvSolMean, tempSurvSolVar, percentChange]], columns = ['R', 'Time', 'Perf Mean', 'Perf Var', '% Improvement'])
            sideExpStats = pd.concat([sideExpStats,tempSideExpStats], ignore_index=True)
    
            if computer == 0:
                sideExpStats.to_excel(f"/home/ephouser/RSP/Results/SideExp/SideExpStats_{searchType}_{dataInstance}.xlsx", index = False)
            else:
                sideExpStats.to_excel(f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\SideExpStats_{searchType}_{dataInstance}.xlsx", index = False)
            
            ###########################################################################################################################
    
            ###############################
            # Define New Incumbent Solution
            tempSurvSolMeanPrevious = tempSurvSolMean
            
            ###########################################################################################################################
                
        if r>0:             
            
            # Getting Variable Selection Ranking
            screenSol = []
            for i in IPrime[-1]:
                #if i > b[0]:
                screenSol.append(X[i])
            
            screenDF = pd.DataFrame(screenSol)
            
            tempSummary = []
            for i in range(len(screenDF.columns)):
                count = sum(screenDF.iloc[:,i])
                tempSummary.append([dataX_Train_MacJ.columns[i] ,count])
            
            names = ["Variable", "Times Selected"]
            tempVarSummary = pd.DataFrame(tempSummary, columns = names).sort_values(by=["Times Selected"], ascending = False, ignore_index = True)
            
            if r == 1:
                positionHistory["Variable"] = dataX_Train_MacJ.columns
            
            # Get Position
            for varIDX in range(len(dataX_Train_MacJ.columns)):
                currentIdx = tempVarSummary[tempVarSummary["Variable"] == dataX_Train_MacJ.columns[varIDX]].index.tolist()[0] + 1
                positionHistory.loc[varIDX,r] = currentIdx
                
            tempAvg = []
            tempVar = []
            for var_i in tempVarSummary["Variable"].values:
                
                varRow = positionHistory.loc[positionHistory["Variable"].values == var_i]
                
                if len(varRow.iloc[0,1:]) > 1:
                    varAvg = round(varRow.iloc[0,1:].mean(),3)
                    varVar = round(varRow.iloc[0,1:].var(),3)
                else:
                    varAvg = varRow.iloc[0,1:].mean()
                    varVar = varRow.iloc[0,1:].var()
                    
                tempAvg.append(varAvg)
                tempVar.append(varVar)
            
            varSummary = tempVarSummary
            varSummary["Average"] = tempAvg
            varSummary["Variance"] = tempVar
            
            varSummary = varSummary.sort_values(by=["Times Selected", "Average"], ascending = [False, True], ignore_index = True)
            
            #############################################################################################################################
            
            if EXP_K == "RSP":
            
                #########################################
                #########################################
                ### SIDE EXP: Store Feat Ranks by ITR ###
                #########################################
                #########################################
                # Side Experiment Ranks By Frequency of Selection    
                sideExpVarSummary = varSummary.sort_values(by=["Times Selected"], ascending = False, ignore_index = True)
                sideExpRanksFrequency[r] = sideExpVarSummary['Variable']
                    
                # Side Experiment Ranks By Position Average    
                sideExpVarSummary = varSummary.sort_values(by=["Average"], ascending = True, ignore_index = True)
                sideExpRanksAverage[r] = sideExpVarSummary['Variable']
                
                #############
                # Export Data
                if computer == 0:
                    sideExpRanksFrequency.to_excel(f"/home/ephouser/RSP/Results/SideExp/SideExpRanks_Frequency_{searchType}_{dataInstance}.xlsx")
                    sideExpRanksAverage.to_excel(f"/home/ephouser/RSP/Results/SideExp/SideExpRanks_Average_{searchType}_{dataInstance}.xlsx")   
                else:
                    sideExpRanksFrequency.to_excel(f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\SideExpRanks_Frequency_{searchType}_{dataInstance}.xlsx")
                    sideExpRanksAverage.to_excel(f"G:\\.shortcut-targets-by-id\\18ebo_mDN_a4hqrRIkCCLM08ghy_hrh5f\\Ethan\\Code\\RSP\\Results\\SideExp\\SideExpRanks_Average_{searchType}_{dataInstance}.xlsx")                
              

            #############################################################################################################################
            
            #################
            ### Print Results
            print("Top 20 Variables")
            print(varSummary.head(20))
            
            if EXP_K == "RSP":
            
                print(" ")
                print(f"% Change in Survivor Mean = {round(percentChange,4)}")
                print(f"Improvement Count = {countImprovement}")
                print(" ")
            
            print(f"Completed Iteration: Dataset = {dataSize} | Macro = {EXP_J_STR} | R = {r} | M = {M}")
            # print('i1 = ', i1, " / ", len(seed_I1))
            
            # if budgetLeft == False:
            #     print("BUDGET EXCEEDED")
            #     break
        
        print(f"Solutions Visited: {B[-1]}")
        print(f"iSR: {iSR}")
        print("\n")
        
        if countImprovement == 5:
            break
            
    # print(Ybari)
    
    print(f"{len(X)} / {2**d} = {len(X)/(2**d)} solutions were explored.")
    
    if EXP_K == 'RSP':

        tempAvg = []
        tempVar = []
        for i in range(len(positionHistory["Variable"])):
            tempAvg.append(positionHistory.iloc[i,1:].mean())
            tempVar.append(positionHistory.iloc[i,1:].var())
        
        if isinstance(tempAvg, (int, float)):
            positionHistory["Average"] = round(tempAvg,3)        
        if isinstance(tempVar, (int, float)):
            positionHistory["Variance"] = round(tempVar,3) 
        
        return varSummary, iSR, dataTrainValidMic, positionHistory, modelName
        
    if EXP_K == 'ORS':
        
        ##################################
        ## PHASE 2: STOPPING AND SELECTION
        r = R+1
        
        ####################################
        ## GET NUMBER OF SURVIVING SOLUTIONS
        numSurvSol = len(IPrime[-1])

        ###########################################
        ## IF ONE SOLUTION SURVIVES RAPID SCREENING        
        if numSurvSol == 1:
            
            #################################################
            ## RETURN REMAINING SOLUTION AS THE BEST SOLUTION
            for i in IPrime[-1]:
                
                ##############
                # GET SOLUTION
                bestSol = np.array(X[i])
                finalFeats = list(dataTrainValidMic[0]['trainX'].loc[:,bestSol==1].columns)

        ################################################
        ## IF MULTIPLE SOLUTIONS SURVIVE RAPID SCREENING        
        if numSurvSol > 1:
            
            ###########################
            # COMPUTE RINOTT'S CONSTANT
            h = getRinotth(n = numSurvSol, r0 = max(Ni.iloc[-2, :]), pstar = 1-alpha2, conf = 0.99, rep = 10000)
            
            ######################################################################
            # COMPUTE THE TOTAL NUMBER OF REPLICATIONS FOR EACH SURVIVING SOLUTION
            for i in range(len(IPrime[-1])):
                idx = IPrime[-1][i]
                
                Si2 = (1 / (Ni.iloc[-2,idx] - 1)) * sum((Yij.iloc[-2, idx] - Ybari_SurvSol[i])**2)
                Ni.iloc[-1,idx] = min(int(max(n[-1], math.ceil((((h**2)*(Si2))/(deltaS**2))))),10000)
            
            ##########################################
            # CALCULATE SURVIVING SOLUTION PERFORMANCE
            bestSolPerf = -9999999
            bestSol = []
            Ybari_Final = []
            for i in IPrime[-1]:
                
                ##############
                # GET SOLUTION
                tempSurvSol = np.array(X[i])
                
                ##########################
                # IF SOLUTION IS ALL ZEROS
                if sum(tempSurvSol) == 0:            
                    Yij.iloc[-1,i] = np.array([-1*99999 for iVal in range(n[-1])])        
                    tempYbari_Final = -1*99999
                    Ybari_Final.append(tempYbari_Final)
                    
                else:
                    if modelType == 0:
                        
                        if Ni.iloc[-1,i] > len(dataTrainValidMic):
                            dataTrainValidMic = getMoreTrainTestData(dataX_Train_MacJ, dataY_Train_MacJ,testSize, 
                                                                      int(Ni.iloc[-1,i]) - len(dataTrainValidMic), seedTrainTest, dataTrainValidMic)                    
                        
                        iSR += Ni.iloc[-1,i]
                        
                        tempYij = []                    
                        for j in range(int(Ni.iloc[-1,i])):
                            temp_trainX = dataTrainValidMic[j]['trainX']
                            temp_testX = dataTrainValidMic[j]['testX']
                            temp_trainY = dataTrainValidMic[j]['trainY']
                            temp_testY  = dataTrainValidMic[j]['testY']
                            
                            tempMAE, tempMSE, tempRMSE = fitLinearReg(temp_trainX.loc[:,tempSurvSol==1],
                                                                      temp_testX.loc[:,tempSurvSol==1],
                                                                      temp_trainY,
                                                                      temp_testY)
                            tempYij.append(-1*tempRMSE)
                        
                        if i <= len(Yij.columns):
                            Yij.iloc[-1,i] = list(np.array(tempYij))                    
                        else:
                            tempYijDF = pd.DataFrame(np.zeros((R+2, i - len(Yij.columns))))
                            Yij = pd.concat([Yij, tempYijDF], axis = 1)
                            Yij.iloc[-1,i] = list(np.array(tempYij))    
                        
                        tempYbari_Final = round((1/Ni.iloc[-1,i])*sum(Yij.iloc[-1,i]),5)
                        Ybari_Final.append(tempYbari_Final)
                          
                    if tempYbari_Final > bestSolPerf:
                        bestSolPerf = tempYbari_Final
                        bestSol = tempSurvSol
            
            finalFeats = list(temp_trainX.loc[:,bestSol==1].columns)
            
            
        return finalFeats, iSR, dataTrainValidMic, [], modelName























