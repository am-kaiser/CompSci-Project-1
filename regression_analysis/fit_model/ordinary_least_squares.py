"""script to calculate ordinary least squares"""

from regression_analysis.utils import create_data_franke
from regression_analysis.utils import basis_functionality
from regression_analysis.utils import create_plots
from regression_analysis.utils import prepare_data

import numpy as np
import matplotlib.pyplot as plt


# To make results comparable set seed()
np.random.seed(2021)


def calculate_OLS(input_x1, input_x2, obs_var, order, D=None, beta_OLS=None):
    """Calculate the response variable for ordinary least squares."""
    if not(D):
        # Get design matrix
        regress_obj = basis_functionality.Design_Matrix_2D(input_x1, input_x2, obs_var, order)
        D = regress_obj.make_design_matrix()
        A_inv = regress_obj.design_matrix_product_inverse()

    if not(beta_OLS):
        # Calculate parameter vector
        beta_OLS = np.dot(np.dot(A_inv, D.T), obs_var.flatten())

    # Calculate response variable
    resp_var_OLS = np.dot(D, beta_OLS).reshape(obs_var.shape[0], obs_var.shape[1])

    return beta_OLS, resp_var_OLS


def perform_OLS(input_x1, input_x2, obs_var, order, train_frac, cross_val=False, num_fold=None, bootstrap=False):
    """Performs ordinary least squares for a 2D function using polynomials of order "order".
        The train ratio is the fraction of data that is used as training data.
        Additional parameter enables bootstrap resampling and cross_validation.
    """
    # Step1: Scale data
    input_x1 = prepare_data.scale_data(input_x1)
    input_x2 = prepare_data.scale_data(input_x2)
    obs_var = prepare_data.scale_data(obs_var)

    # Step 2: Perform sampling
    if cross_val:
        # Perform cross-validation
        x1_train, x2_train, x1_test, x2_test, y_train, y_test = prepare_data.cross_validation(input_x1, input_x2,
                                                                                              obs_var, num_fold)
    elif bootstrap:
        # Split data in test and train datasets
        x1_train, x2_train, x1_test, x2_test, y_train, y_test = prepare_data.split_test_train(input_x1, input_x2,
                                                                                              obs_var,
                                                                                              train_fraction=train_frac,
                                                                                              do_shuffle=False)
        # Perform bootstrap
        x1_train, x2_train, y_train = prepare_data.bootstrap(x1_train, x2_train, y_train)
    else:
        # Split data in test and train datasets
        x1_train, x2_train, x1_test, x2_test, y_train, y_test = prepare_data.split_test_train(input_x1, input_x2,
                                                                                              obs_var,
                                                                                              train_fraction=train_frac,
                                                                                              do_shuffle=True)

    # Step 3: Calculate OLS
    # Calculate beta and response variable for train dataset
    beta_OLS_train, resp_var_OLS_train = calculate_OLS(x1_train, x2_train, y_train, order)
    # Calculate error evaluaters for the train dataset
    error_class = basis_functionality.Error_Measures(y_train, resp_var_OLS_train)
    MSE_train = error_class.mean_squared_error()
    R2_train = error_class.r2_score()

    # Get OLS response variable for test dataset
    regress_obj_test = basis_functionality.Design_Matrix_2D(x1_test, x2_test, y_test, order)
    D_test = regress_obj_test.make_design_matrix()
    resp_var_OLS_test = np.dot(D_test, beta_OLS_train).reshape(y_test.shape[0], y_test.shape[1])

    # Step 4: Calculate errors
    # Calculate error evaluaters for the test dataset
    error_class = basis_functionality.Error_Measures(y_test, resp_var_OLS_test)
    MSE_test = error_class.mean_squared_error()
    R2_test = error_class.r2_score()

    return beta_OLS_train, resp_var_OLS_train, resp_var_OLS_test, MSE_train, MSE_test, R2_train, R2_test


if __name__ == "__main__":

    # Get data
    input_x1, input_x2, obs_var = create_data_franke.generate_data(noisy=False, noise_variance=0.5, uniform=False,
                                                                   points=100)

    # Fit model for different polynomial
    max_order = 5
    orders = range(1, max_order+1)
    MSE_train = np.empty([1, max_order])
    R2_train = np.empty([1, max_order])
    MSE_test = np.empty([1, max_order])
    R2_test = np.empty([1, max_order])

    for i, order in enumerate(orders):
        _, _, _, MSE_train[:, i], MSE_test[:, i], R2_train[:, i], R2_test[:, i] = perform_OLS(input_x1, input_x2,
                                                                                              obs_var, order,
                                                                                              train_frac=0.8,
                                                                                              cross_val=True,
                                                                                              num_fold=5,
                                                                                              bootstrap=False)

    # Plot errors
    axes_1 = np.array(orders)
    args = (MSE_train, MSE_test, R2_train, R2_test)
    axes_2 = np.concatenate(args, axis=0)

    line_lab = ['MSE train', 'MSE test', 'R2 train', 'R2 test']

    fig_errors = plt.figure()
    create_plots.make_multi_line_plot(axes_1, axes_2[:2, :], line_lab[:2], fig_errors, 211)
    create_plots.make_multi_line_plot(axes_1, axes_2[2:, :], line_lab[2:], fig_errors, 212)

    plt.show()
