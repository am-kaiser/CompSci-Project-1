"""Unit test for the contents of regression_analysis.fit_model.linear_regression."""
import numpy as np

from regression_analysis.fit_model import linear_regression
from regression_analysis.utils import franke


def test_poly_powers2D():
    """Test if the correct order of polynomials is produced."""
    x1_powers, x2_powers = linear_regression.poly_powers2D(2)

    expected_pow = np.array([0, 0, 0, 1, 1, 2])

    np.testing.assert_array_equal(np.sort(x1_powers), np.sort(x2_powers))
    np.testing.assert_array_equal(np.sort(x1_powers), expected_pow)


def test_design_mat2D():
    pass


def test_find_model_parameter():
    """Test if the betas from our code is the same as the ones from scikit-learn."""
    # Get data from Franke function
    x1, x2, y = franke.create_data(num_points=100, noise_variance=0)
    # Get design matrix
    X = linear_regression.design_mat2D(np.squeeze(x1), np.squeeze(x2), 5)
    # Get beta from own code
    beta_OLS_own = linear_regression.find_model_parameter(X, y, "ols", 0)
    beta_RR_own = linear_regression.find_model_parameter(X, y, "ridge", 1)
    # Get beta from scikit
    beta_OLS_scikit = linear_regression.find_model_parameter(X, y, "scikit_ols", 0)
    beta_RR_scikit = linear_regression.find_model_parameter(X, y, "scikit_ridge", 1)

    # Compare results from own code to scikit-learn
    np.testing.assert_array_almost_equal(beta_OLS_own, beta_OLS_scikit, decimal=6)
    np.testing.assert_array_almost_equal(beta_RR_own, beta_RR_scikit, decimal=6)
