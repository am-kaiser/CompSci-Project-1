"""Script to apply stochastic gradient descend to ordinary least squares and ridge regression."""

import numpy as np

from regression_analysis.utils import create_data_franke, basis_functionality


def gradient_RR_OLS(y, D, b, lam):
    """
    Defines the gradient for ordinary least squares and ridge gression.
    When lam=0 it is OLS and otherwise ridge regression.
    """
    n = len(y)
    gradient = (-2/n)*D.T@(y-D@b) + 2*lam*b
    return gradient


def stochastic_gradient_descent_method(gradient, obs_var, matrix, start, num_epoch, learn_rate, num_min_batch, labda):
    """Define gradient descent method to find optimal beta for given gradient."""
    vector = start
    n = matrix.shape[0]
    for _ in range(num_epoch):
        for _ in range(num_min_batch):
            batch_index = np.random.randint(n, size=int(n/num_min_batch))
            matrix_batch = matrix[batch_index, :]
            obs_var_batch = obs_var[batch_index]
            descend = learn_rate*gradient(y=obs_var_batch, D=matrix_batch, b=vector, lam=labda)
            # Stop if all values are smaller or equal than machine precision
            if np.all(descend) <= np.finfo(float).eps:
                break
            vector -= descend
    return vector


if __name__ == "__main__":

    # Get data
    input_x1, input_x2, obs_var_f = create_data_franke.generate_data(noisy=False, noise_variance=0.5, uniform=False,
                                                                     points=100)

    regress_obj = basis_functionality.Design_Matrix_2D(input_x1, input_x2, obs_var_f, order=5)
    design_matrix = regress_obj.make_design_matrix()

    beta_OLS = stochastic_gradient_descent_method(gradient=gradient_RR_OLS, obs_var=obs_var_f.flatten(), matrix=design_matrix,
                                                  start=np.repeat(0.0, design_matrix.shape[1]), num_epoch=1, learn_rate=0.1, num_min_batch=1, labda=0)

    beta_RR = stochastic_gradient_descent_method(gradient=gradient_RR_OLS, obs_var=obs_var_f.flatten(), matrix=design_matrix,
                                                 start=np.repeat(0.0, design_matrix.shape[1]), num_epoch=1, learn_rate=0.1, num_min_batch=1,
                                                 labda=0)
