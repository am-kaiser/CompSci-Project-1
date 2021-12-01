"""Script to perform logistic regression."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_name='regression_analysis/examples/data_logistic_regression/data.csv'):
    """Load data, select the column with diagnosis and transform content to {0,1}."""
    data = pd.read_csv(file_name, sep=',')
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    return data


def design_matrix(data):
    """Create design matrix from data."""
    return data.drop(['id', 'diagnosis'], axis=1).values


def normalise_data(matrix):
    """Normalise given matrix"""
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def apply_logistic_regression(self, test_ratio=0.1):
        if test_ratio != 0.0:
            # Split data ind training and testing datasets
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


if __name__ == "__main__":
    input_data = load_data()
    X_obs = normalise_data(design_matrix(input_data))
    y_in = input_data.diagnosis.values
    print(type(X_obs), y_in)
