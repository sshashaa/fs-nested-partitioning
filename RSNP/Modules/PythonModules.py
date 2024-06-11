# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:57:00 2024

@author: ephouser
"""

##################################
## RSP Driver
import numpy as np
import pandas as pd
from datetime import *
import random, math, sys, os, copy, scipy.stats, itertools
from itertools import combinations
import os.path
import ast

pd.options.mode.chained_assignment = None

##################################
## Estimating Performance
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, fbeta_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

##################################
## Generating Datasets
from sklearn.model_selection import train_test_split

##################################
## Load Datasets

##################################
## NP Solution Search
import scipy, math

##################################
## Parallel Processing
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed


##################################
## Rapid Screening
import matplotlib.pyplot as plt

##################################
## Rapid Screening Solution Search
    
# Get the current time with proper format
def getTimeNow():
    myTime = datetime.now().strftime("%H:%M:%S")
    myTime = datetime.strptime(myTime, "%H:%M:%S")
    return myTime