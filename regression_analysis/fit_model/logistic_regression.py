"""Script to perform logistic regression or support vector machine algorithms."""

import os

import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn.model_selection import train_test_split

from regression_analysis.utils import stochastic_gradient_descent, sampling


def load_data(file_name='regression_analysis/examples/data_logistic_regression/data.csv'):
    """Load data, select the column with diagnosis and transform content to {0,1}."""
    current_path = os.getcwd()
    current_directory = current_path[current_path.rindex(os.sep) + 1:]
    if current_directory == 'examples':
        file_name = 'data_logistic_regression/data.csv'
    else:
        file_name = 'regression_analysis/examples/data_logistic_regression/data.csv'

    data = pd.read_csv(file_name, sep=',')
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    return data


def design_matrix(data):
    """Create design matrix from data."""
    data_val = data.drop(['id', 'diagnosis'], axis=1).values
    intercept = np.repeat(1, data.shape[0])
    return np.c_[intercept, data_val]


def normalise_data(matrix):
    """Normalise given matrix column-wise"""
    norm_matrix = np.zeros([matrix.shape[0], matrix.shape[1]])
    norm_matrix[:, 0] = matrix[:, 0]
    for col in np.arange(1, matrix.shape[1]):
        denominator = np.max(matrix[:, col]) - np.min(matrix[:, col])
        if denominator != 0:
            norm_matrix[:, col] = (matrix[:, col] - np.min(matrix[:, col])) / denominator
        else:
            norm_matrix[:, col] = np.repeat(1.0, matrix.shape[0])

    return norm_matrix


def find_logistic_model_parameter(design_matrix, y_input, method, lam, num_epoch, learn_rate, num_min_batch):
    """
    Calculate beta for given method
    :param design_matrix: design matrix
    :param y_input: y from training data
    :param method: fitting method to be used
    :param lam: L2 regularization parameter
    :param num_epoch: number of epochs for stochastic gradient descent
    :param learn_rate: learn rate for stochastic gradient descent
    :param num_min_batch: number of mini batches for stochastic gradient descent
    :return: beta/ parameters
    """
    if method == "logistic_sgd":
        start_vec = np.repeat(0.0, design_matrix.shape[1])
        gradient = stochastic_gradient_descent.gradient_LR
        parameter = stochastic_gradient_descent.stochastic_gradient_descent_method(gradient, y=y_input, X=design_matrix, start=start_vec,
                                                                                   num_epoch=num_epoch, learn_rate=learn_rate,
                                                                                   num_min_batch=num_min_batch, lmbda=lam)
    elif method == "logistic_scikit":
        parameter = linear_model.LogisticRegression(fit_intercept=False).fit(design_matrix, np.ravel(y_input)).coef_.T  # penalty???
    elif method == "svm":
        parameter = svm.SVC(kernel='linear').fit(design_matrix, np.ravel(y_input))

    return parameter


def make_confusion_matrix(y, y_predict):
    """Create confusion matrix for y and prediction of y."""
    true_pos = np.count_nonzero((y == y_predict) & (y == 1))
    true_neg = np.count_nonzero((y == y_predict) & (y == 0))
    false_pos = np.count_nonzero((y != y_predict) & (y == 0))
    false_neg = np.count_nonzero((y != y_predict) & (y == 1))

    confusion_matrix = np.array([[true_pos, false_neg], [false_pos, true_neg]])

    return pd.DataFrame(confusion_matrix, index=['is_malignant', 'is_benign'], columns=['predicted_malignent', 'predicted_benign'])


def transform_0_1(values):
    """Transform content of numpy array to either 0 or 1."""
    return np.where(values < 0, 0, 1)


class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(y.shape[0], 1)

        self.train_accuracy = np.nan
        self.test_accuracy = np.nan
        self.train_confusion_matrix = pd.DataFrame()
        self.test_confusion_matrix = pd.DataFrame()

    def apply_logistic_regression(self, test_ratio=None, reg_method=None, lmbda=None, num_epoch=None, learn_rate=None, num_min_batch=None):
        """
        Perform logistic regression
        :param test_ratio: ratio of data used as a test dataset
        :param reg_method: fitting method to be used
        :param lmbda: L2 regularization parameter
        :param num_epoch: number of epochs for stochastic gradient descent
        :param learn_rate: learn rate for stochastic gradient descent
        :param num_min_batch: number of mini batches for stochastic gradient descent
        :return: accuracy and confusion matrix for training and testing datasets
        """
        if test_ratio != 0.0:
            # Split data in training and testing datasets
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_ratio)
        else:
            x_train = self.X
            x_test = np.array([])
            y_train = self.y
            y_test = np.array([])

        if reg_method == 'svm':
            # Find model parameters
            svm = find_logistic_model_parameter(x_train, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
            # Predict using training data
            y_model_train = svm.predict(x_train)[:, np.newaxis]
            # Calculate accuracy for training data
            self.train_accuracy = np.mean(y_train == y_model_train)
            # Calculate confusion matrix
            self.train_confusion_matrix = make_confusion_matrix(y_train, y_model_train)

            if test_ratio != 0.0:
                # Predict using testing data
                y_model_test = svm.predict(x_test)[:, np.newaxis]
                # Calculate accuracy for training data
                self.test_accuracy = np.mean(y_test == y_model_test)
                # Calculate confusion matrix
                self.test_confusion_matrix = make_confusion_matrix(y_test, y_model_test)
        else:
            # Find model parameters
            beta = find_logistic_model_parameter(x_train, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
            # Fit training data
            y_model_train = np.array(x_train @ beta)
            # transform y_model_train to be either 0 or 1
            y_model_train = transform_0_1(y_model_train)
            # Calculate accuracy for training data
            self.train_accuracy = np.mean(y_train == y_model_train)
            # Calculate confusion matrix
            self.train_confusion_matrix = make_confusion_matrix(y_train, y_model_train)

            if test_ratio != 0.0:
                # Fit model to testing data
                y_model_test = np.array(x_test @ beta)
                # transform y_model_test to be either 0 or 1
                y_model_test = transform_0_1(y_model_test)
                # Calculate accuracy for training data
                self.test_accuracy = np.mean(y_test == y_model_test)
                # Calculate confusion matrix
                self.test_confusion_matrix = make_confusion_matrix(y_test, y_model_test)

    def apply_logistic_regression_crossvalidation(self, kfolds=None, reg_method=None, lmbda=None, num_epoch=None, learn_rate=None,
                                                  num_min_batch=None):
        """
        Perform logistic regression with k fold cross validation resampling
        :param kfolds: number of folds to be used with cross-validation
        :param reg_method: fitting method to be used
        :param lmbda: L2 regularization parameter
        :param num_epoch: number of epochs for stochastic gradient descent
        :param learn_rate: learn rate for stochastic gradient descent
        :param num_min_batch: number of mini batches for stochastic gradient descent
        :return: accuracy and confusion matrix for training and testing datasets
        """
        [self.train_accuracy, self.test_accuracy] = [0.0, 0.0]

        x_train_arr, x_test_arr, y_train_arr, y_test_arr = sampling.crossvalidation(self.X, self.y, kfolds)

        for k in np.arange(kfolds):
            x_train = x_train_arr[k, :, :]
            x_test = x_test_arr[k, :, :]

            y_train = y_train_arr[k, :].reshape(len(y_train_arr[k, :]), 1)
            y_test = y_test_arr[k, :].reshape(len(y_test_arr[k, :]), 1)

            if reg_method == 'svm':
                # finding model parameters
                svm = find_logistic_model_parameter(x_train, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
                # Fit model for test and train data
                y_model_test = svm.predict(x_test)[:, np.newaxis]
                y_model_train = svm.predict(x_train)[:, np.newaxis]
            else:
                # finding model parameters
                beta = find_logistic_model_parameter(x_train, y_train, reg_method, lmbda, num_epoch, learn_rate, num_min_batch)
                # Fit model for test and train data
                y_model_test = np.array(x_test @ beta)
                y_model_train = np.array(x_train @ beta)
                # transform y's to be either 0 or 1
                y_model_test = transform_0_1(y_model_test)
                y_model_train = transform_0_1(y_model_train)
            # Calculate error statistics
            self.train_accuracy += np.mean(y_train == y_model_train)
            self.test_accuracy += np.mean(y_test == y_model_test)
            # Calculate confusion matrix
            self.train_confusion_matrix = self.train_confusion_matrix.add(make_confusion_matrix(y_train, y_model_train), fill_value=0)
            self.test_confusion_matrix = self.test_confusion_matrix.add(make_confusion_matrix(y_test, y_model_test), fill_value=0)

        # Calculate mean of each error statistic
        self.train_accuracy /= kfolds
        self.test_accuracy /= kfolds

        self.train_confusion_matrix = (self.train_confusion_matrix / kfolds).round(0).astype('int')
        self.test_confusion_matrix = (self.test_confusion_matrix / kfolds).round(0).astype('int')
