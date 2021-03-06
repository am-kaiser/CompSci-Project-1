import numpy as np


def findMSE(y_data, y_fit):
    """Calculate mean squared error of input."""
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_fit), 1)
    n = len(y_data)
    if n == 0:
        return np.nan  # edge case for zero testing data
    return np.mean((y_data - y_fit) ** 2)


def findR2(y_data, y_fit):
    """Calculate R2 score of input."""
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_fit), 1)
    if len(y_data) == 0:
        return np.nan  # edge case for zero testing data
    num = np.sum((y_data - y_fit) ** 2)
    den = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - (num / den)


def findBias(y_data, y_fit):
    """Calculate bias of input."""
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_fit), 1)
    y_mean = np.mean(y_fit)
    return np.mean((y_data - y_mean) ** 2)


def findModelVar(y_fit):
    """Calculate variance of input."""
    return np.var(y_fit)
