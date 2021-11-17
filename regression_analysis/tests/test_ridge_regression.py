"""Unit test for the contents of regression_analysis.ridge_regression module."""

from regression_analysis.fit_model import ridge_regression
import numpy as np
from sklearn import linear_model


def test_correct_beta():
    # Intercept is included in the design matrix
    design_matrix = np.array([[1, 3, 2], [1, 5, 4], [1, 7, 8], [1, 9, 10]])
    obs_var = np.array([10, 3, 7, 15]).reshape(4, 1)
    alpha = 0.5

    # Calculate beta with scikit-learn
    RR_reg_sklearn = linear_model.Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix, obs_var)
    beta_sklearn = RR_reg_sklearn.coef_

    # Calculate beta with own code
    beta_own, _ = ridge_regression.calculate_RR(design_matrix[:, 2], design_matrix[:, 1], obs_var, 1, alpha)

    assert beta_sklearn.all() == beta_own.all()
