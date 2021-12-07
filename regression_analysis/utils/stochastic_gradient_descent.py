"""Script to use ordinary least squares and ridge regression with stochastic gradient descend."""

import numpy as np


def gradient_RR_OLS(y, X, beta, lmbda):
    """
    Define the gradient for ordinary least squares and ridge regression.
    :params y: observed values
    :params X: design matrix
    :params beta: parameter vector/ regression coefficients
    :params lmbda: When lam=0 it is OLS and otherwise ridge regression.
    :return: gradient of cost function
    """
    n = len(y)
    gradient = (-2 / n) * X.T @ (y - X @ beta) + 2 * lmbda * beta
    return gradient


def sigmoid_func(z):
    return 1 / (1 + np.exp(-z))


def gradient_LR(y, X, beta, lmbda):
    """
    Define the gradient for logistic regression.
    :params y: observed values
    :params X: design matrix
    :params beta: parameter vector/ regression coefficients
    :params lmbda: L2 regularization parameter
    :return: gradient of cost function
    """
    gradient = (-1) * X.T @ (y - sigmoid_func(X @ beta)) - lmbda*beta
    return gradient


def stochastic_gradient_descent_method(gradient, y, X, start, num_epoch, learn_rate, num_min_batch, lmbda):
    """
    Define gradient descent method to find optimal beta for given gradient.
    :params gradient: gradient of cost function
    :params y: observed values
    :params X: design matrix
    :params start: initial values
    :params num_epoch: number of epochs
    :params learn_rate: learn rate
    :params num_min_batch. number of mini batches
    :params lmbda: When lam=0 it is OLS and otherwise ridge regression.
    :return: beta
    """
    vector = start.reshape(start.shape[0], 1)
    num_observations = X.shape[0]
    for _ in range(num_epoch):
        for _ in range(num_min_batch):
            batch_index = np.random.randint(num_observations, size=int(num_observations / num_min_batch))
            X_batch = X[batch_index, :]
            y_batch = y[batch_index]
            descend = learn_rate * gradient(y=y_batch, X=X_batch, beta=vector, lmbda=lmbda)
            # Stop if all values are smaller or equal than machine precision
            if np.all(descend) <= np.finfo(float).eps:
                break
            vector -= descend
    return vector
