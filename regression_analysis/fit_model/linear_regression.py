import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from regression_analysis.utils import sampling, findStat


def poly_powers2D(order):
    """
    determines the array of powers of the two dependent variables
    for a 2D polynomial
    input: order of 2D polynomial
    output: arrays with powers for both dependent variables
    """
    x1pow = [0]
    x2pow = [0]
    for i in np.arange(1, order + 1):
        for j in np.arange(0, i + 1):
            x1pow.append(j)
            x2pow.append(i - j)
    return x1pow, x2pow


def design_mat2D(x1, x2, order):
    """
    Generate design matrix for 2D Least Squares
    input: arrays of input features, order of polynomial
    output: design matrix
    """
    x1pow, x2pow = poly_powers2D(order)
    n = np.size(x1)
    design_mat = np.zeros((n, len(x1pow)))
    for term in range(len(x1pow)):
        design_mat[:, term] = (x1 ** x1pow[term]) * (x2 ** x2pow[term])
    return design_mat


def find_ols_params(X, y_train):
    """
    find beta aka parameters for ordinary least squares
    Input: Design matrix, training output
    output: beta
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y_train  # beta


def find_ridge_params(X, y_train, lmbda):
    """
    find beta aka parameters for ridge least squares
    Input: Design matrix, training output, lambda
    output: beta
    """
    identity = np.eye(X.shape[1], X.shape[1])
    return np.linalg.pinv(X.T @ X + lmbda * identity) @ X.T @ y_train  # beta


class linear_regression2D():
    def __init__(self, x1, x2, y, **kwargs):
        """
        initialise data for regression
        """
        self.n_points = y.shape[0]
        # fixing data dimensions
        if (len(x1.shape) == 1 or len(x1.shape) == 1 or len(x1.shape) == 1):
            x1 = x1.reshape(self.n_points, 1)
            x2 = x2.reshape(self.n_points, 1)
            y = y.reshape(self.n_points, 1)

        self.x1 = x1  # input 2Ddata
        self.x2 = x2
        self.y = y  # output
        self.n_points = y.shape[0]
        self.trainMSE = np.nan
        self.trainR2 = np.nan
        self.testMSE = np.nan
        self.testR2 = np.nan
        self.trainbias = np.nan
        self.testbias = np.nan
        self.trainvar = np.nan
        self.testvar = np.nan

        # scaling data using mix max scaling
        self.y = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))

    def apply_leastsquares(self, order=3, test_ratio=0.1, reg_method="ols", lmbda=0.1):
        """
        performs least squares
        input: order of polynomial, test ratio
        optional inputs: "ridge" -> if True, performs ridge regression with
        lambda = lmbda
        """
        if (test_ratio != 0.0):
            x_train, x_test, y_train, y_test = train_test_split(np.hstack([self.x1, self.x2]), self.y,
                                                                test_size=test_ratio)

            x1_train = x_train[:, 0]
            x2_train = x_train[:, 1]

            x1_test = x_test[:, 0]
            x2_test = x_test[:, 1]
        else:
            x1_train = self.x1.flatten()
            x2_train = self.x2.flatten()
            x1_test = np.array([])
            x2_test = np.array([])
            y_train = self.y
            y_test = np.array([])

        # find train design matrix
        X = design_mat2D(x1_train, x2_train, order)

        # finding model parameters
        if reg_method == "ols":
            beta = find_ols_params(X, y_train)
        elif reg_method == "ridge":
            beta = find_ridge_params(X, y_train, lmbda)
        elif reg_method == "scikit_ols":
            beta = linear_model.LinearRegression(fit_intercept=False).fit(X, y_train).coef_.T
        elif reg_method == "scikit_ridge":
            beta = linear_model.Ridge(alpha=lmbda, fit_intercept=False).fit(X, y_train).coef_.T
        elif reg_method == "scikit_lasso":
            beta = linear_model.Lasso(alpha=lmbda, fit_intercept=False).fit(X, y_train).coef_.T
            
        y_model_train = np.array(X @ beta)  # fitting training data

        self.trainMSE = findStat.findMSE(y_train, y_model_train)
        self.trainR2 = findStat.findR2(y_train, y_model_train)
        self.trainbias = findStat.findBias(y_train, y_model_train)
        self.trainvar = findStat.findModelVar(y_model_train)

        if (test_ratio != 0.0):
            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)  # fitting testing data
            self.testMSE = findStat.findMSE(y_test, y_model_test)
            self.testR2 = findStat.findR2(y_test, y_model_test)
            self.testbias = findStat.findBias(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)

    def apply_leastsquares_bootstrap(self, order=3, test_ratio=0.1, n_boots=10, reg_method="ols", lmbda=0.1):
        """
        performs least squares with bootstrap sampling
        input: order of polynomial, test ratio, number of bootstraps
        optional inputs: "ridge" -> if True, performs ridge regression with
        lambda = lmbda
        """
        [self.trainMSE, self.trainR2, self.testMSE, self.testR2] = [0.0, 0.0, 0.0, 0.0]
        for run in range(n_boots):
            x_train, x_test, y_train, y_test = sampling.bootstrap(np.hstack([self.x1, self.x2]), self.y,
                                                                  sample_ratio=test_ratio)
            x1_train = x_train[:, 0]
            x2_train = x_train[:, 1]

            x1_test = x_test[:, 0]
            x2_test = x_test[:, 1]

            # find train design matrix
            X = design_mat2D(x1_train, x2_train, order)
            # finding model parameters
            if reg_method == "ols":
                beta = find_ols_params(X, y_train)
            elif reg_method == "ridge":
                beta = find_ridge_params(X, y_train, lmbda)
            elif reg_method == "scikit_ols":
                beta = linear_model.LinearRegression(fit_intercept=False).fit(X, y_train).coef_.T
            elif reg_method == "scikit_ridge":
                beta = linear_model.Ridge(alpha=lmbda, fit_intercept=False).fit(X, y_train).coef_.T
            elif reg_method == "scikit_lasso":
                beta = linear_model.Lasso(alpha=lmbda, fit_intercept=False).fit(X, y_train).coef_.T

            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)  # fitting testing data
            y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta)  # fitting training data

            self.trainMSE += findStat.findMSE(y_train, y_model_train)
            self.trainR2 += findStat.findR2(y_train, y_model_train)
            self.testMSE += findStat.findMSE(y_test, y_model_test)
            self.testR2 += findStat.findR2(y_test, y_model_test)
            self.trainbias = findStat.findBias(y_train, y_model_train)
            self.trainvar = findStat.findModelVar(y_model_train)
            self.testbias = findStat.findBias(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)
        self.trainMSE /= n_boots
        self.testMSE /= n_boots
        self.trainR2 /= n_boots
        self.testR2 /= n_boots
        self.trainbias /= n_boots
        self.testbias /= n_boots
        self.trainvar /= n_boots
        self.testvar /= n_boots

    def apply_leastsquares_crossvalidation(self, order=3, kfolds=10, reg_method="ols", lmbda=0.1):
        """
        performs least squares with k fold cross validation sampling
        input: order of polynomial, test ratio, number of folds
        optional inputs: "ridge" -> if True, performs ridge regression with
        lambda = lmbda
        """
        [self.trainMSE, self.trainR2, self.testMSE, self.testR2] = [0.0, 0.0, 0.0, 0.0]

        x_train_arr, x_test_arr, y_train_arr, y_test_arr = sampling.crossvalidation(np.hstack([self.x1, self.x2]),
                                                                                    self.y, kfolds)

        for k in np.arange(kfolds):
            x1_train = x_train_arr[k, :, 0]
            x2_train = x_train_arr[k, :, 1]

            x1_test = x_test_arr[k, :, 0]
            x2_test = x_test_arr[k, :, 1]

            y_train = y_train_arr[k, :].reshape(len(y_train_arr[k, :]), 1)
            y_test = y_test_arr[k, :].reshape(len(y_test_arr[k, :]), 1)

            # find train design matrix
            X = design_mat2D(x1_train, x2_train, order)
            # finding model parameters
            if reg_method == "ols":
                beta = find_ols_params(X, y_train)
            elif reg_method == "ridge":
                beta = find_ridge_params(X, y_train, lmbda)
            elif reg_method == "scikit_ols":
                beta = linear_model.LinearRegression(fit_intercept=False).fit(X, y_train).coef_
            elif reg_method == "scikit_ridge":
                beta = linear_model.Ridge(alpha=lmbda, fit_intercept=False).fit(X, y_train).coef_
            elif reg_method == "scikit_lasso":
                beta = linear_model.Lasso(alpha=lmbda, fit_intercept=False).fit(X, y_train).coef_

            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)  # fitting testing data
            y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta)  # fitting training data

            self.trainMSE += findStat.findMSE(y_train, y_model_train)
            self.trainR2 += findStat.findR2(y_train, y_model_train)
            self.testMSE += findStat.findMSE(y_test, y_model_test)
            self.testR2 += findStat.findR2(y_test, y_model_test)
            self.trainbias = findStat.findBias(y_train, y_model_train)
            self.trainvar = findStat.findModelVar(y_model_train)
            self.testbias = findStat.findBias(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)
        self.trainMSE /= kfolds
        self.testMSE /= kfolds
        self.trainR2 /= kfolds
        self.testR2 /= kfolds
        self.trainbias /= kfolds
        self.testbias /= kfolds
        self.trainvar /= kfolds
        self.testvar /= kfolds
