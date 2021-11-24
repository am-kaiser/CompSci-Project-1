import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import findStat
import sampling

def poly_powers2D(order):
    x1pow = [0]
    x2pow = [0]
    for i in np.arange(1,order+1):
        for j in np.arange(0,i+1):
            x1pow.append(j)
            x2pow.append(i-j)
    return x1pow, x2pow

def design_mat2D(x1, x2, order):
    x1pow, x2pow = poly_powers2D(order)
    n = np.size(x1)
    design_mat = np.zeros((n, len(x1pow)))
    for term in range(len(x1pow)):
        design_mat[:, term] = (x1**x1pow[term])*(x2**x2pow[term])
    return design_mat

def find_params(X, y_train):
    X = np.asmatrix(X)
    XT = np.asmatrix(np.transpose(X))
    XTX = XT*X
    C = np.linalg.inv(XTX)
    beta = (C*XT)*y_train #model parameters
    return beta

def ols2D(x1, x2, y, order, test_ratio=0.0, bootstrap=False, n_boots = 10, cross_validation=False, kfolds=5):
    """performs ordinary least squares for a 2D function using polynomials of order "order".
        The test ratio is the fraction of data that is used as testing data. 
        training and testing: MSE and R2 errors are the outputs

        Additional parameters enables bootstrap and cross validation resampling. 
        n_boots is the number of times we sample using the bootstrap method.
        kfolds is the number of folds in cross validation sampling
    """
    points = np.size(x1)

    #splitting data
    x = np.zeros((points, 2))

    if(test_ratio!=0.0):
        if not bootstrap: #fitting without resampling
            x_train, x_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, test_size=test_ratio)
            
            x1_train = x_train[:, 0]
            x2_train = x_train[:, 1]

            x1_test = x_test[:, 0]
            x2_test = x_test[:, 1]

            #find train design matrix
            X = design_mat2D(x1_train, x2_train, order)

            #finding model parameters
            beta = find_params(X, y_train)

            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta) #fitting testing data
            y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta) #fitting training data

            trainMSE = findStat.findMSE(y_train, y_model_train)
            trainR2 = findStat.findR2(y_train, y_model_train)
            testMSE = findStat.findMSE(y_test, y_model_test)
            testR2 = findStat.findR2(y_test, y_model_test)
        elif bootstrap:
            [trainMSE, trainR2, testMSE, testR2] = [0.0, 0.0, 0.0, 0.0]
            for run in n_boots:
                x_train, x_test, y_train, y_test = sampling.bootstrap(np.hstack([x1, x2]), y, sample_ratio=test_ratio)
                x1_train = x_train[:, 0]
                x2_train = x_train[:, 1]

                x1_test = x_test[:, 0]
                x2_test = x_test[:, 1]

                #find train design matrix
                X = design_mat2D(x1_train, x2_train, order)

                #finding model parameters
                beta = find_params(X, y_train)

                y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta) #fitting testing data
                y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta) #fitting training data

                trainMSE += findStat.findMSE(y_train, y_model_train)
                trainR2 += findStat.findR2(y_train, y_model_train)
                testMSE += findStat.findMSE(y_test, y_model_test)
                testR2 += findStat.findR2(y_test, y_model_test)
            trainMSE /= n_boots
            testMSE /= n_boots
            trainR2 /= n_boots
            testR2 /= n_boots


"""
    if(test_ratio!= 0.0):
        if not (bootstrap and cross_validation):
            x_train, x_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, test_size=test_ratio)   
        elif bootstrap:
            print("bootstraping")
            x_train, x_test, y_train, y_test = sampling.bootstrap(np.hstack([x1, x2]), y, sample_ratio=test_ratio)
        elif cross_validation:
            #to dox_train, x_test, y_train, y_test = sampling.bootstrap(np.hstack([x1, x2]), y, sample_ratio=test_ratio)
            print("to do")

        x1_train = x_train[:, 0]
        x2_train = x_train[:, 1]

        x1_test = x_test[:, 0]
        x2_test = x_test[:, 1]

    else:
        x1_train = x1.flatten()
        x2_train = x2.flatten()
        x1_test = np.array([])
        x2_test = np.array([])
        y_train = y
        y_test = np.array([])

    #find train design matrix
    X = design_mat2D(x1_train, x2_train, order)

    #finding model parameters
    beta = find_params(X, y_train)

    y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta) #fitting testing data
    y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta) #fitting training data

    trainMSE = findStat.findMSE(y_train, y_model_train)
    trainR2 = findStat.findR2(y_train, y_model_train)
    if(test_ratio != 0.0):
        testMSE = findStat.findMSE(y_test, y_model_test)
        testR2 = findStat.findR2(y_test, y_model_test)
    else: 
        #no testing data available
        testMSE = np.nan
        testR2 = np.nan

    return trainMSE, testMSE, trainR2, testR2


"""