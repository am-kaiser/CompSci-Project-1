"""Scripts to perform fitting different least square methods to resampled data."""
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from regression_analysis.utils import sampling, findStat, stochastic_gradient_descent


def poly_powers2D(order):
    """
    Determines the array of powers of the two dependent variables for a 2D polynomial.
    :param order: order of polynomial to be fitted
    :return: arrays with powers for both dependent variables
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
    Generate design matrix for 2D least squares.
    :param x1: array of input features
    :param x2: array of input features
    :param order: order of polynomial to be fitted
    :return: design matrix
    """
    x1pow, x2pow = poly_powers2D(order)
    n = np.size(x1)
    design_mat = np.zeros((n, len(x1pow)))
    for term in range(len(x1pow)):
        design_mat[:, term] = (x1 ** x1pow[term]) * (x2 ** x2pow[term])
    return design_mat


def find_ols_params(X, y_train):
    """
    Find beta aka parameters for ordinary least squares.
    :param X: design matrix
    :param y_train: y from training data
    :return: beta/ parameters
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y_train


def find_ridge_params(X, y_train, lmbda):
    """
    Find beta aka parameters for ridge regression.
    :param X: design matrix
    :param y_train: y from training data
    :param lmbda: lambda for ridge or lasso regression
    :return: beta/ parameters
    """
    identity = np.eye(X.shape[1], X.shape[1])
    return np.linalg.pinv(X.T @ X + lmbda * identity) @ X.T @ y_train


def find_model_parameter(design_matrix, y_input, method, lam, num_epoch, learn_rate, num_min_batch):
    """
    Calculate beta for given method.
    :param design_matrix: design matrix
    :param y_input: y from training data
    :param method: fitting method to be used
    :param lam: lambda for ridge or lasso regression
    :params num_epoch: number of epochs for stochastic gradient descent
    :params learn_rate: learn rate for stochastic gradient descent
    :params num_min_batch. number of mini batches for stochastic gradient descent
    :return: beta/ parameters
    """
    if method == "ols":
        parameter = find_ols_params(design_matrix, y_input)
    elif method == "ridge":
        parameter = find_ridge_params(design_matrix, y_input, lam)
    elif method == "scikit_ols":
        parameter = linear_model.LinearRegression(fit_intercept=False).fit(design_matrix, y_input).coef_.T
    elif method == "scikit_ridge":
        parameter = linear_model.Ridge(alpha=lam, fit_intercept=False).fit(design_matrix, y_input).coef_.T
    elif method == "scikit_lasso":
        parameter = linear_model.Lasso(alpha=lam, fit_intercept=False).fit(design_matrix, y_input).coef_.T
    elif method == "ols_sgd":
        start_vec = np.repeat(0.0, design_matrix.shape[1])
        gradient = stochastic_gradient_descent.gradient_RR_OLS
        parameter = stochastic_gradient_descent.stochastic_gradient_descent_method(gradient, y=y_input, X=design_matrix, start=start_vec,
                                                                                   num_epoch=num_epoch, learn_rate=learn_rate,
                                                                                   num_min_batch=num_min_batch, lmbda=0)
    elif method == "ridge_sgd":
        start_vec = np.repeat(0.0, design_matrix.shape[1])
        gradient = stochastic_gradient_descent.gradient_RR_OLS
        parameter = stochastic_gradient_descent.stochastic_gradient_descent_method(gradient, y=y_input, X=design_matrix, start=start_vec,
                                                                                   num_epoch=num_epoch, learn_rate=learn_rate,
                                                                                   num_min_batch=num_min_batch, lmbda=lam)
    return parameter


class linear_regression2D():
    def __init__(self, x1, x2, y, **kwargs):
        """Initialise data for regression"""
        self.n_points = y.shape[0]
        # Fixing data dimensions
        if len(x1.shape) == 1 or len(x2.shape) == 1 or len(y.shape) == 1:
            x1 = x1.reshape(self.n_points, 1)
            x2 = x2.reshape(self.n_points, 1)
            y = y.reshape(self.n_points, 1)

        self.x1 = x1  # input 2D data
        self.x2 = x2
        self.y = y
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

    def apply_leastsquares(self, order=3, test_ratio=0.1, reg_method="ols", lmbda=0.1, num_epoch=50, learn_rate=0.1, num_min_batch=5):
        """
        Performs least squares on training and testing data.
        :param order: order of polynomial which will be fitted
        :param test_ratio: size of testing data set
        :param reg_method: fitting method to be used
        :param lmbda: lambda for ridge or lasso regression
        :params num_epoch: number of epochs for stochastic gradient descent
        :params learn_rate: learn rate for stochastic gradient descent
        :params num_min_batch. number of mini batches for stochastic gradient descent
        :return: MSE, R2, bias, variance for training and testing datasets
        """
        if test_ratio != 0.0:
            # Split data ind training and testing datasets
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

        # Find train design matrix
        X = design_mat2D(x1_train, x2_train, order)
        # Find model parameters
        beta = find_model_parameter(X, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
        # Fit training data
        y_model_train = np.array(X @ beta)

        # Calculate error statistics for training data
        self.trainMSE = findStat.findMSE(y_train, y_model_train)
        self.trainR2 = findStat.findR2(y_train, y_model_train)
        self.trainbias = findStat.findBias(y_train, y_model_train)
        self.trainvar = findStat.findModelVar(y_model_train)

        if test_ratio != 0.0:
            # Fit model to testing data
            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)
            # Calculate error statistics for testing data
            self.testMSE = findStat.findMSE(y_test, y_model_test)
            self.testR2 = findStat.findR2(y_test, y_model_test)
            self.testbias = findStat.findBias4(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)
            # self.testbias = findStat.findBias3(y_train, y_test, y_model_train, y_model_test)

    def apply_leastsquares_bootstrap(self, order=3, test_ratio=0.1, n_boots=10, reg_method="ols", lmbda=0.1, num_epoch=50, learn_rate=0.1,
                                     num_min_batch=5):
        """
        Performs least squares with bootstrap resampling.
        :param order: order of polynomial which will be fitted
        :param test_ratio: size of testing data set
        :param n_boots: number of bootstraps to be performed
        :param reg_method: fitting method to be used
        :param lmbda: lambda for ridge or lasso regression
        :params num_epoch: number of epochs for stochastic gradient descent
        :params learn_rate: learn rate for stochastic gradient descent
        :params num_min_batch. number of mini batches for stochastic gradient descent
        :return: MSE, R2, bias, variance for training and testing datasets
        """
        [self.trainMSE, self.trainR2, self.testMSE, self.testR2] = [0.0, 0.0, 0.0, 0.0]
        for run in range(n_boots):
            x_train, x_test, y_train, y_test = sampling.bootstrap(np.hstack([self.x1, self.x2]), self.y,
                                                                  sample_ratio=test_ratio)
            x1_train = x_train[:, 0]
            x2_train = x_train[:, 1]

            x1_test = x_test[:, 0]
            x2_test = x_test[:, 1]

            # Find train design matrix
            X = design_mat2D(x1_train, x2_train, order)
            # Find model parameters
            beta = find_model_parameter(X, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
            # Fit model for test and train data
            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)
            y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta)
            # Calculate error statistics
            self.trainMSE += findStat.findMSE(y_train, y_model_train)
            self.trainR2 += findStat.findR2(y_train, y_model_train)
            self.testMSE += findStat.findMSE(y_test, y_model_test)
            self.testR2 += findStat.findR2(y_test, y_model_test)
            self.trainbias = findStat.findBias(y_train, y_model_train)
            self.trainvar = findStat.findModelVar(y_model_train)
            self.testbias = findStat.findBias(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)
        # Calculate mean of each error statistic
        self.trainMSE /= n_boots
        self.testMSE /= n_boots
        self.trainR2 /= n_boots
        self.testR2 /= n_boots
        self.trainbias /= n_boots
        self.testbias /= n_boots
        self.trainvar /= n_boots
        self.testvar /= n_boots

    def apply_leastsquares_crossvalidation(self, order=3, kfolds=10, reg_method="ols", lmbda=0.1, num_epoch=50, learn_rate=0.1,
                                           num_min_batch=5):
        """
        Perform least squares with k fold cross validation resampling.
        :param order: order of polynomial which will be fitted
        :param kfolds: number of folds to be used with cross-validation
        :param reg_method: fitting method to be used
        :param lmbda: lambda for ridge or lasso regression
        :return: MSE, R2, bias, variance for training and testing datasets
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
            beta = find_model_parameter(X, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
            # Fit model for test and train data
            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)
            y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta)
            # Calculate error statistics
            self.trainMSE += findStat.findMSE(y_train, y_model_train)
            self.trainR2 += findStat.findR2(y_train, y_model_train)
            self.testMSE += findStat.findMSE(y_test, y_model_test)
            self.testR2 += findStat.findR2(y_test, y_model_test)
            self.trainbias = findStat.findBias(y_train, y_model_train)
            self.trainvar = findStat.findModelVar(y_model_train)
            self.testbias = findStat.findBias(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)
        # Calculate mean of each error statistic
        self.trainMSE /= kfolds
        self.testMSE /= kfolds
        self.trainR2 /= kfolds
        self.testR2 /= kfolds
        self.trainbias /= kfolds
        self.testbias /= kfolds
        self.trainvar /= kfolds
        self.testvar /= kfolds


if __name__ == "__main__":
    from regression_analysis.utils import franke

    x1, x2, y = franke.create_data(num_points=100, noise_variance=0)

    model = linear_regression2D(x1, x2, y)
    model.apply_leastsquares(order=5, test_ratio=0.1, reg_method="ols")
