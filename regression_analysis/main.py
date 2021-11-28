import numpy as np
import matplotlib.pyplot as plt
import ordinaryLeastSquares
import findStat
import franke
import linear_regression
from numpy import random as npr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
## recheck bootstrap
##parallel
## flake 8 or some linter

if __name__ == '__main__':
    
    n = 103
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
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

    n = 100
    x1 = np.linspace(0,1,n)
    x2 = np.linspace(0,1,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = franke.Franke(xx1, xx2, var=0.7)

    x_train, x_test, y_train, y_test = train_test_split(np.hstack([xx1, xx2]), y, test_size=0.2)
            
    x1_train = x_train[:, 0]
    x2_train = x_train[:, 1]

    x1_test = x_test[:, 0]
    x2_test = x_test[:, 1]
    """
    x1_train = xx1[:,0]
    x2_train = xx2[:,0]
    y_train = y
    """
    #comparing with SKlearn results
    X = linear_regression.design_mat2D(x1_train, x2_train, order=5)
    OLS_reg_sklearn = LinearRegression(fit_intercept=False).fit(X, y_train)
    beta_sklearn = OLS_reg_sklearn.coef_.T
    print(beta_sklearn.shape)
    print(X.shape)
    ytrain_fit = X@beta_sklearn
    Xtest = linear_regression.design_mat2D(x1_test, x2_test, order=5)
    ytest_fit = Xtest@beta_sklearn
    trainMSE = findStat.findMSE(y_train, ytrain_fit)
    testMSE = findStat.findMSE(y_test, ytest_fit)
    print(testMSE)
    print(trainMSE)

    #lasso scikit
    lasso_reg = linear_model.Lasso(alpha=0.000001, fit_intercept=False)
    lasso_reg.fit(X, y_train)
    ytrain_fit = X@lasso_reg.coef_
    ytest_fit = Xtest@lasso_reg.coef_
    testR2 = findStat.findR2(y_test, ytest_fit)
    print(ytrain_fit)
    print(ytest_fit)
    print(testR2)




