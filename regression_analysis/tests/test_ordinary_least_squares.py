"""Unit test for the contents of regression_analysis.ordinary_least_squares module."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from regression_analysis.fit_model import ordinary_least_squares
from regression_analysis.utils import basis_functionality


def test_correct_beta():
    # Intercept is included in the design matrix
    design_matrix = np.array([[1, 3, 2], [1, 5, 4], [1, 7, 8], [1, 9, 10]])
    obs_var = np.array([10, 3, 7, 15]).reshape(4, 1)

    # Calculate beta with scikit-learn
    OLS_reg_sklearn = LinearRegression(fit_intercept=False).fit(design_matrix, obs_var)
    beta_sklearn = OLS_reg_sklearn.coef_
    # Calculate beta with own code
    beta_OLS_own, _ = ordinary_least_squares.calculate_OLS(design_matrix[:, 2], design_matrix[:, 1], obs_var=obs_var,
                                                           order=1)

    assert beta_sklearn.all() == beta_OLS_own.all()


def test_errors():
    # Intercept is included in the design matrix
    design_matrix = np.array([[1, 3, 2], [1, 5, 4], [1, 7, 8], [1, 9, 10]])
    obs_var = np.array([10, 3, 7, 15]).reshape(4, 1)

    # Calculate response variable with own code
    _, resp_var_OLS = ordinary_least_squares.calculate_OLS(design_matrix[:, 2], design_matrix[:, 1], obs_var=obs_var,
                                                           order=1)

    # Calculate mean squared error with scikit-learn
    MSE_sklearn = mean_squared_error(obs_var, resp_var_OLS)

    # Calculate mean squared error with own code
    error_class = basis_functionality.Error_Measures(obs_var, resp_var_OLS)
    MSE_own = error_class.mean_squared_error()

    assert MSE_sklearn == MSE_own
