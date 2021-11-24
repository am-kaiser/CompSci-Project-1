import numpy as np
import matplotlib.pyplot as plt
import ordinaryLeastSquares
import findStat
import franke
import linear_regression
from numpy import random as npr
from sklearn.linear_model import LinearRegression

## recheck bootstrap

if __name__ == '__main__':
    
    n = 103
    x1 = np.linspace(0,1,n)
    x2 = np.linspace(0,1,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = franke.Franke(xx1, xx2, var=0)
    print(y.shape)
    
    """
    n = 100
    x1 = np.linspace(1,n,n)
    x2 = np.linspace(1,n,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = xx1+xx2**2    
    """

    linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)
    linear_reg.apply_leastsquares(order=5, test_ratio=0.1)
    print("*********")
    print("OLS Results")
    print("Train MSE", linear_reg.trainMSE)
    print("Test MSE", linear_reg.testMSE)
    linear_reg.apply_leastsquares_bootstrap(order=5, test_ratio=0.1, n_boots=10)
    print("*********")
    print("OLS with bootstrap Results")
    print("Train MSE", linear_reg.trainMSE)
    print("Test MSE", linear_reg.testMSE)

    linear_reg.apply_leastsquares_crossvalidation(order=5, kfolds=10)
    print("*********")
    print("OLS with cross validation Results")
    print("Train MSE", linear_reg.trainMSE)
    print("Test MSE", linear_reg.testMSE)

    linear_reg.apply_leastsquares(order=5, test_ratio=0.1, ridge=True, lmbda=0.1)
    print("*********")
    print("Ridge LS Results")
    print("Train MSE", linear_reg.trainMSE)
    print("Test MSE", linear_reg.testMSE)

    linear_reg.apply_leastsquares_bootstrap(order=5, test_ratio=0.1, n_boots=10, ridge=True, lmbda=0.1)
    print("*********")
    print("Ridge LS with bootstrap Results")
    print("Train MSE", linear_reg.trainMSE)
    print("Test MSE", linear_reg.testMSE)

    linear_reg.apply_leastsquares_crossvalidation(order=5, kfolds=10, ridge=True, lmbda=0.1)
    print("*********")
    print("Ridge LS with cross validation Results")
    print("Train MSE", linear_reg.trainMSE)
    print("Test MSE", linear_reg.testMSE)

    xx1 = np.array([3,5,7,9])
    xx2 = np.array([2,4,8,10])
    y = np.array([10,3,7,15])

    linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)
    linear_reg.apply_leastsquares(order=2, test_ratio=0.0)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_leastsquares_bootstrap(order=1, test_ratio=0.1, n_boots=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_leastsquares_crossvalidation(order=1, kfolds=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    print(linear_reg.trainbias)
    print(linear_reg.testbias)
    #comparing with SKlearn results
    X = linear_regression.design_mat2D(xx1, xx2, order=2)
    OLS_reg_sklearn = LinearRegression(fit_intercept=False).fit(X, y)
    beta_sklearn = OLS_reg_sklearn.coef_
    #y_fit = beta_sklearn@X

    x1 = np.arange(0,10)
    x2 = np.arange(0,10)




