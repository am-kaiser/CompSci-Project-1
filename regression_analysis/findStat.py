import numpy as np

def findMSE(y_data, y_fit):
    n = len(y_data)
    if(n==0):
        return np.nan #edge case for zero testing data
    return np.sum((y_data-y_fit)**2)/n

def findR2(y_data, y_fit):
    if(len(y_data)==0):
        return np.nan #edge case for zero testing data
    num = np.sum((y_data-y_fit)**2)
    den = np.sum((y_data-np.mean(y_data))**2)
    return 1.0-(num/den)

def findBias(y_data, y_fit):
    y_mean = np.mean(y_fit)
    return np.sum((y_data-y_mean)**2)

def findModelVar(y_fit):
    return np.var(y_fit)