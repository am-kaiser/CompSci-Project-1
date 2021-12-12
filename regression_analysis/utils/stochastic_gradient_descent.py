"""Script to use ordinary least squares and ridge regression with stochastic gradient descend."""

import numpy as np


def gradient_RR_OLS(y, X, beta, lmbda):
    """
    Define the gradient for ordinary least squares and ridge regression
    :param y: observed values
    :param X: design matrix
    :param beta: parameter vector/ regression coefficients
    :param lmbda: When lam=0 it is OLS and otherwise ridge regression.
    :return: gradient of cost function
    """
    n = len(y)
    gradient = (-2 / n) * X.T @ (y - X @ beta) + 2 * lmbda * beta
    return gradient


def sigmoid_func(z):
    """Calculate sigmoid function."""
    sigmoid = np.zeros((z.shape[0], 1))
    for elem_in, elem in enumerate(z):
        if -745 <= elem <= 745:
            sigmoid[elem_in, 0] = 1 / (1 + np.exp(-z[elem_in, 0]))
        elif elem > 745:
            sigmoid[elem_in, 0] = 1  # exp(-z) -> 0 if z -> inf
        else:
            sigmoid[elem_in, 0] = 0  # exp(-z) -> inf if z -> -inf and 1/inf = 0

    return sigmoid


def gradient_LR(y, X, beta, lmbda):
    """
    Define the gradient for logistic regression
    :param y: observed values
    :param X: design matrix
    :param beta: parameter vector/ regression coefficients
    :param lmbda: L2 regularization parameter
    :return: gradient of cost function
    """
    gradient = (-1) * X.T @ (y - sigmoid_func(X @ beta)) - lmbda*beta
    return gradient


def stochastic_gradient_descent_method(gradient, y, X, start, num_epoch, learn_rate, num_min_batch, lmbda):
    """
    Define gradient descent method to find optimal beta for given gradient
    :param gradient: gradient of cost function
    :param y: observed values
    :param X: design matrix
    :param start: initial values
    :param num_epoch: number of epochs
    :param learn_rate: learn rate
    :param num_min_batch. number of mini batches
    :param lmbda: When lam=0 it is OLS and otherwise ridge regression.
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
            if np.any(descend) <= np.finfo(float).eps:
                break
            vector -= descend
    return vector
