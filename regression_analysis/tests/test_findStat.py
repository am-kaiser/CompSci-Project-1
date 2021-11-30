"""Unit test for the contents of regression_analysis.utils.findStat."""

import numpy as np
import sklearn.metrics as sm

from regression_analysis.fit_model import linear_regression
from regression_analysis.utils import franke, findStat


def get_model_output():
    # Get data from Franke function
    x1, x2, y = franke.create_data(num_points=100, noise_variance=0)
    # Get design matrix
    X = linear_regression.design_mat2D(np.squeeze(x1), np.squeeze(x2), 5)
    # Get betas from scikit
    beta_OLS = linear_regression.find_model_parameter(X, y, "scikit_ols", 0)
    beta_RR = linear_regression.find_model_parameter(X, y, "scikit_ridge", 1)
    beta_LR = linear_regression.find_model_parameter(X, y, "scikit_lasso", 1)
    # Get y
    y_OLS = X @ beta_OLS
    y_RR = X @ beta_RR
    y_LR = X @ beta_LR

    return y, y_OLS, y_RR, y_LR


def test_findMSE():
    """Test if the MSE from our code is the same as the one from scikit-learn."""
    y, y_OLS, y_RR, y_LR = get_model_output()
    # Calculate mean squared error
    MSE_OLS_own = findStat.findMSE(y, y_OLS)
    MSE_RR_own = findStat.findMSE(y, y_RR)
    MSE_LR_own = findStat.findMSE(y, y_LR)
    MSE_OLS_scikit = sm.mean_squared_error(y, y_OLS)
    MSE_RR_scikit = sm.mean_squared_error(y, y_RR)
    MSE_LR_scikit = sm.mean_squared_error(y, y_LR)

    np.testing.assert_array_equal(MSE_OLS_own, MSE_OLS_scikit)
    np.testing.assert_array_equal(MSE_RR_own, MSE_RR_scikit)
    np.testing.assert_array_equal(MSE_LR_own, MSE_LR_scikit)


def test_findR2():
    """Test if the R2 score from our code is the same as the one from scikit-learn."""
    y, y_OLS, y_RR, y_LR = get_model_output()
    # Calculate mean squared error
    R2_OLS_own = findStat.findR2(y, y_OLS)
    R2_RR_own = findStat.findR2(y, y_RR)
    R2_LR_own = findStat.findR2(y, y_LR)
    R2_OLS_scikit = sm.r2_score(y, y_OLS)
    R2_RR_scikit = sm.r2_score(y, y_RR)
    R2_LR_scikit = sm.r2_score(y, y_LR)

    np.testing.assert_array_equal(R2_OLS_own, R2_OLS_scikit)
    np.testing.assert_array_equal(R2_RR_own, R2_RR_scikit)
    np.testing.assert_array_equal(R2_LR_own, R2_LR_scikit)
