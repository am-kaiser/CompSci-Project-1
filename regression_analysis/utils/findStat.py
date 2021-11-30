import numpy as np


def findMSE(y_data, y_fit):
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_data), 1)
    n = len(y_data)
    if n == 0:
        return np.nan  # edge case for zero testing data
    return np.mean((y_data - y_fit) ** 2)


def findR2(y_data, y_fit):
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_data), 1)
    if len(y_data) == 0:
        return np.nan  # edge case for zero testing data
    num = np.sum((y_data - y_fit) ** 2)
    den = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - (num / den)


def findBias(y_data, y_fit):
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_data), 1)
    y_mean = np.mean(y_fit)
    return np.mean((y_data - y_mean) ** 2)


def findModelVar(y_fit):
    return np.var(y_fit)


def findBias2(y_train, y_test, y_trainfit, y_testfit):
    y = np.vstack([y_train, y_test])
    y_fit = np.vstack([y_trainfit, y_testfit])
    y_mean = np.mean(y_fit)
    return np.mean((y - y_mean) ** 2)


def findBias3(y_train, y_test, y_trainfit, y_testfit):
    y = np.vstack([y_train, y_test])
    y_fit = np.vstack([y_trainfit, y_testfit])
    y_mean = np.mean(y_fit)
    return np.mean((y - y_mean))


def findBias4(y_data, y_fit):
    y_data = y_data.reshape(len(y_data), 1)
    y_fit = y_fit.reshape(len(y_data), 1)
    y_mean = np.mean(y_fit)
    return np.mean((y_data - y_mean))
