"""Script to apply stochastic gradient descend to ordinary least squares and ridge regression."""

import numpy as np

from regression_analysis.utils import create_data_franke, basis_functionality


def gradient_RR_OLS(y, D, beta, lam):
    """
    Defines the gradient for ordinary least squares and ridge gression.
    When lam=0 it is OLS and otherwise ridge regression.
    """
    n = len(y)
    gradient = (-2/n)*D.T@(y-D@beta) + 2*lam*beta
    return gradient


def stochastic_gradient_descent_method(gradient, y, D, start, num_epoch, learn_rate, num_min_batch, lam):
    """Define gradient descent method to find optimal beta for given gradient."""
    beta = start
    n = D.shape[0]
    for _ in range(num_epoch):
        for _ in range(num_min_batch):
            batch_index = np.random.randint(n, size=int(n/num_min_batch))
            D_batch = D[batch_index, :]
            y_batch = y[batch_index]
            descend = learn_rate*gradient(y_batch, D_batch, beta, lam)
            # Stop if all values are smaller or equal than machine precision
            if np.all(descend) <= np.finfo(float).eps:
                break
            beta -= descend
    return beta


if __name__ == "__main__":

    # Get data
    input_x1, input_x2, obs_var = create_data_franke.generate_data(noisy=False, noise_variance=0.5, uniform=False,
                                                                   points=100)

    regress_obj = basis_functionality.Design_Matrix_2D(input_x1, input_x2, obs_var, order=5)
    design_matrix = regress_obj.make_design_matrix()
    start_vec = np.repeat(0.0, design_matrix.shape[1])
    """
    beta_RR = stochastic_gradient_descent_method(gradient=gradient_RR_OLS, y=obs_var.flatten(), D=design_matrix,
                                                 start=start_vec, num_epoch=1, learn_rate=0.1, num_min_batch=1,
                                                 lam=90)"""
    beta_OLS = stochastic_gradient_descent_method(gradient=gradient_RR_OLS, y=obs_var.flatten(), D=design_matrix,
                                                  start=start_vec, num_epoch=1, learn_rate=0.1, num_min_batch=1, lam=0)

    print(beta_OLS)
    print(beta_RR)
