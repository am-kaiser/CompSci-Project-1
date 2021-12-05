"""Script to perform logistic regression."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from regression_analysis.utils import stochastic_gradient_descent, findStat, sampling


def load_data(file_name='regression_analysis/examples/data_logistic_regression/data.csv'):
    """Load data, select the column with diagnosis and transform content to {0,1}."""
    data = pd.read_csv(file_name, sep=',')
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    return data


def design_matrix(data):
    """Create design matrix from data."""
    intercept = np.repeat(1, data.shape[0])
    data['intercept'] = intercept
    return data.drop(['id', 'diagnosis'], axis=1).values


def normalise_data(matrix):
    """Normalise given matrix"""
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


def find_logistic_model_parameter(design_matrix, y_input, method, lam, num_epoch, learn_rate, num_min_batch):
    """
    Calculate beta for given method.
    :param design_matrix: design matrix
    :param y_input: y from training data
    :param method: fitting method to be used
    :param lam: L2 regularization parameter
    :params num_epoch: number of epochs for stochastic gradient descent
    :params learn_rate: learn rate for stochastic gradient descent
    :params num_min_batch. number of mini batches for stochastic gradient descent
    :return: beta/ parameters
    """
    if method == "logistic_sgd":
        start_vec = np.repeat(0.0, design_matrix.shape[1])
        gradient = stochastic_gradient_descent.gradient_LR
        parameter = stochastic_gradient_descent.stochastic_gradient_descent_method(gradient, y=y_input, X=design_matrix, start=start_vec,
                                                                                   num_epoch=num_epoch, learn_rate=learn_rate,
                                                                                   num_min_batch=num_min_batch, lmbda=lam)
    elif method == "logistic_scikit":
        parameter=LogisticRegression(fit_intercept=False).fit(design_matrix, y_input).coef_.T #penalty???

    return parameter


class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.trainMSE = np.nan
        self.trainR2 = np.nan
        self.testMSE = np.nan
        self.testR2 = np.nan
        self.trainbias = np.nan
        self.testbias = np.nan
        self.trainvar = np.nan
        self.testvar = np.nan

    def apply_logistic_regression(self, test_ratio=0.1, reg_method="logistic_sgd", lmbda=0, num_epoch=50, learn_rate=0.1, num_min_batch=5):
        if test_ratio != 0.0:
            # Split data in training and testing datasets
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_ratio)
            x_train = x_train.T
            x_test = x_test.T
            y_train = y_train.T
            y_test = y_test.T
        else:
            x_train = self.X
            x_test = np.array([])
            y_train = self.y
            y_test = np.array([])

        # Find model parameters
        beta = find_logistic_model_parameter(x_train, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
        # Fit training data
        y_model_train = np.array(x_train @ beta)

        # Calculate error statistics for training data
        self.trainMSE = findStat.findMSE(y_train, y_model_train)
        self.trainR2 = findStat.findR2(y_train, y_model_train)
        self.trainbias = findStat.findBias(y_train, y_model_train)
        self.trainvar = findStat.findModelVar(y_model_train)

        if test_ratio != 0.0:
            # Fit model to testing data
            y_model_test = np.array(x_test @ beta)
            # Calculate error statistics for testing data
            self.testMSE = findStat.findMSE(y_test, y_model_test)
            self.testR2 = findStat.findR2(y_test, y_model_test)
            self.testbias = findStat.findBias4(y_test, y_model_test)
            self.testvar = findStat.findModelVar(y_model_test)
            self.testbias = findStat.findBias3(y_train, y_test, y_model_train, y_model_test)

    def apply_logistsic_regression_crossvalidation(self, kfolds=10, reg_method="logistic_sgd", lmbda=0, num_epoch=50, learn_rate=0.1, num_min_batch=5):
        """
        Perform logistic regression with k fold cross validation resampling.
        :param order: order of polynomial which will be fitted
        :param kfolds: number of folds to be used with cross-validation
        :param reg_method: fitting method to be used
        :param lmbda: L2 regularization parameter
        :return: MSE, R2, bias, variance for training and testing datasets
        """
        [self.trainMSE, self.trainR2, self.testMSE, self.testR2] = [0.0, 0.0, 0.0, 0.0]

        x_train_arr, x_test_arr, y_train_arr, y_test_arr = sampling.crossvalidation(self.X, self.y, kfolds)

        for k in np.arange(kfolds):
            x_train = x_train_arr[k, :, :]
            x_test = x_test_arr[k, :, :]

            y_train = y_train_arr[k, :].reshape(len(y_train_arr[k, :]), 1)
            y_test = y_test_arr[k, :].reshape(len(y_test_arr[k, :]), 1)

            # finding model parameters
            beta = find_logistic_model_parameter(x_train, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
            # Fit model for test and train data
            y_model_test = np.array(x_test @ beta)
            y_model_train = np.array(x_train @ beta)
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
    input_data = load_data()
    X_obs = normalise_data(design_matrix(input_data))
    y_in = input_data.diagnosis.values
    print(X_obs)
