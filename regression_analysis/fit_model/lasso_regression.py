"""Script to calculate Lasso Regression."""

from regression_analysis.utils import create_data_franke
from regression_analysis.utils import basis_functionality
from regression_analysis.utils import create_plots
from regression_analysis.utils import prepare_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def calculate_LR(input_x1, input_x2, obs_var, order, lam, D=None):
    """Calculate the response variable for Ridge Regression. lam stands for lambda."""

    if not(D):
        # Get design matrix
        regress_obj = basis_functionality.Design_Matrix_2D(input_x1, input_x2, obs_var, order)
        D = regress_obj.make_design_matrix()

    model_lasso = linear_model.Lasso(alpha=lam, fit_intercept=False).fit(D, obs_var.flatten())
    beta_LR = model_lasso.coef_
    resp_var_LR = model_lasso.predict(D)

    return beta_LR, resp_var_LR.reshape(obs_var.shape[0], obs_var.shape[1])


def perform_LR(input_x1, input_x2, obs_var, order, lam, train_frac):
    """
    Performs Lasso Regression for a 2D function using polynomials of order "order".
    The train ratio is the fraction of data that is used as training data.
    """
    # Step1: Scale data
    input_x1 = prepare_data.scale_data(input_x1)
    input_x2 = prepare_data.scale_data(input_x2)
    obs_var = prepare_data.scale_data(obs_var)

    # Step 2: Perform sampling
    # Split data in test and train datasets
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = prepare_data.split_test_train(input_x1, input_x2,
                                                                                          obs_var,
                                                                                          train_fraction=train_frac,
                                                                                          do_shuffle=True)

    # Step 3: Calculate Lasso Regression
    # Calculate beta and response variable for train dataset
    beta_LR_train, resp_var_LR_train = calculate_LR(x1_train, x2_train, y_train, order, lam=lam)

    # Calculate error evaluaters for the train dataset
    error_class = basis_functionality.Error_Measures(y_train, resp_var_LR_train)
    MSE_train = error_class.mean_squared_error()
    R2_train = error_class.r2_score()

    # Get Lasso Regression response variable for test dataset
    regress_obj_test = basis_functionality.Design_Matrix_2D(x1_test, x2_test, y_test, order)
    D_test = regress_obj_test.make_design_matrix()
    resp_var_LR_test = np.dot(D_test, beta_LR_train).reshape(y_test.shape[0], y_test.shape[1])

    # Step 4: Calculate errors
    # Calculate error evaluaters for the test dataset
    error_class = basis_functionality.Error_Measures(y_test, resp_var_LR_test)
    MSE_test = error_class.mean_squared_error()
    R2_test = error_class.r2_score()

    return MSE_train, MSE_test, R2_train, R2_test


def perform_LR_cross_val(input_x1, input_x2, obs_var, order, lam, num_fold=5):
    """
    Performs Lasso Regression for a 2D function using polynomials of order "order".
    Additional parameter specifies number of folds for cross_validation.
    """
    # Step1: Scale data
    input_x1 = prepare_data.scale_data(input_x1)
    input_x2 = prepare_data.scale_data(input_x2)
    obs_var = prepare_data.scale_data(obs_var)

    # Step 2: Perform sampling
    # Perform cross-validation

    MSE_train = np.empty([1, num_fold])
    R2_train = np.empty([1, num_fold])
    MSE_test = np.empty([1, num_fold])
    R2_test = np.empty([1, num_fold])

    x_train_arr, x_test_arr, y_train_arr, y_test_arr = prepare_data.cross_validation(input_x1, input_x2, obs_var,
                                                                                     num_fold)

    # Step 3: Calculate LR for each fold
    for fold_index in range(num_fold):

        x1_train = x_train_arr[fold_index, :, 0]
        x2_train = x_train_arr[fold_index, :, 1]

        x1_test = x_test_arr[fold_index, :, 0]
        x2_test = x_test_arr[fold_index, :, 1]

        y_train = y_train_arr[fold_index, :].reshape(len(y_train_arr[fold_index, :]), 1)
        y_test = y_test_arr[fold_index, :].reshape(len(y_test_arr[fold_index, :]), 1)

        # Calculate beta and response variable for train dataset
        beta_LR_train, resp_var_LR_train = calculate_LR(x1_train, x2_train, y_train, order, lam=lam)
        # Calculate error evaluaters for the train dataset
        error_class = basis_functionality.Error_Measures(y_train, resp_var_LR_train)

        MSE_train[0, fold_index] = error_class.mean_squared_error()
        R2_train[0, fold_index] = error_class.r2_score()

        # Get LR response variable for test dataset
        regress_obj_test = basis_functionality.Design_Matrix_2D(x1_test, x2_test, y_test, order)
        D_test = regress_obj_test.make_design_matrix()
        resp_var_LR_test = np.dot(D_test, beta_LR_train).reshape(y_test.shape[0], y_test.shape[1])

        # Step 4: Calculate errors for each fold
        # Calculate error evaluaters for the test dataset
        error_class = basis_functionality.Error_Measures(y_test, resp_var_LR_test)
        MSE_test[0, fold_index] = error_class.mean_squared_error()
        R2_test[0, fold_index] = error_class.r2_score()

    MSE_train_mean = np.mean(MSE_train)
    MSE_test_mean = np.mean(MSE_test)
    R2_train_mean = np.mean(R2_train)
    R2_test_mean = np.mean(R2_test)

    return MSE_train_mean, MSE_test_mean, R2_train_mean, R2_test_mean


def perform_LR_bootstrap(input_x1, input_x2, obs_var, order, lam, train_frac=0.8, num_boot=5):
    """
    Performs lasso regression for a 2D function using polynomials of order "order".
    Additional parameter specifies number of number of times bootstrap will be performed.
    """
    # Step1: Scale data
    input_x1 = prepare_data.scale_data(input_x1)
    input_x2 = prepare_data.scale_data(input_x2)
    obs_var = prepare_data.scale_data(obs_var)

    # Step 2: Perform sampling

    MSE_train = np.empty([1, num_boot])
    R2_train = np.empty([1, num_boot])
    MSE_test = np.empty([1, num_boot])
    R2_test = np.empty([1, num_boot])

    # Split data in test and train datasets
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = prepare_data.split_test_train(input_x1, input_x2,
                                                                                          obs_var,
                                                                                          train_fraction=train_frac,
                                                                                          do_shuffle=True)

    # Step 3: Calculate LR for bootstrap
    for boot_index in range(num_boot):

        # Perform Boostrapping on training data
        x1_train_boot, x2_train_boot, y_train_boot = prepare_data.bootstrap(x1_train, x2_train, y_train)

        # Calculate beta and response variable for train dataset
        beta_LR_train, resp_var_LR_train = calculate_LR(x1_train_boot, x2_train_boot, y_train_boot, order, lam=lam)
        # Calculate error evaluaters for the train dataset
        error_class = basis_functionality.Error_Measures(y_train_boot, resp_var_LR_train)

        MSE_train[0, boot_index] = error_class.mean_squared_error()
        R2_train[0, boot_index] = error_class.r2_score()

        # Get LR response variable for test dataset
        regress_obj_test = basis_functionality.Design_Matrix_2D(x1_test, x2_test, y_test, order)
        D_test = regress_obj_test.make_design_matrix()
        resp_var_LR_test = np.dot(D_test, beta_LR_train).reshape(y_test.shape[0], y_test.shape[1])

        # Step 4: Calculate errors for each fold
        # Calculate error evaluaters for the test dataset
        error_class = basis_functionality.Error_Measures(y_test, resp_var_LR_test)
        MSE_test[0, boot_index] = error_class.mean_squared_error()
        R2_test[0, boot_index] = error_class.r2_score()

    MSE_train_mean = np.mean(MSE_train)
    MSE_test_mean = np.mean(MSE_test)
    R2_train_mean = np.mean(R2_train)
    R2_test_mean = np.mean(R2_test)

    return MSE_train_mean, MSE_test_mean, R2_train_mean, R2_test_mean


if __name__ == "__main__":

    # Get data
    input_x1, input_x2, obs_var = create_data_franke.generate_data(noisy=False, noise_variance=0.5, uniform=False,
                                                                   points=5)

    # Fit model for different polynomial
    max_order = 5
    orders = range(1, max_order+1)

    MSE_train = np.empty([1, max_order])
    R2_train = np.empty([1, max_order])
    MSE_test = np.empty([1, max_order])
    R2_test = np.empty([1, max_order])

    MSE_train_c = np.empty([1, max_order])
    R2_train_c = np.empty([1, max_order])
    MSE_test_c = np.empty([1, max_order])
    R2_test_c = np.empty([1, max_order])

    MSE_train_b = np.empty([1, max_order])
    R2_train_b = np.empty([1, max_order])
    MSE_test_b = np.empty([1, max_order])
    R2_test_b = np.empty([1, max_order])

    for i, order in enumerate(orders):
        MSE_train[:, i], MSE_test[:, i], R2_train[:, i], R2_test[:, i] = perform_LR(input_x1, input_x2, obs_var, order,
                                                                                    lam=2, train_frac=0.8)
        MSE_train_c[:, i], MSE_test_c[:, i], R2_train_c[:, i], R2_test_c[:, i] = perform_LR_cross_val(input_x1,
                                                                                                      input_x2,
                                                                                                      obs_var,
                                                                                                      order,
                                                                                                      lam=2,
                                                                                                      num_fold=5)
        MSE_train_b[:, i], MSE_test_b[:, i], R2_train_b[:, i], R2_test_b[:, i] = perform_LR_bootstrap(input_x1,
                                                                                                      input_x2,
                                                                                                      obs_var,
                                                                                                      order,
                                                                                                      lam=2,
                                                                                                      train_frac=0.8,
                                                                                                      num_boot=5)

    # Plot errors
    axes_1 = np.array(orders)

    args_MSE = (MSE_train, MSE_test, MSE_train_c, MSE_test_c, MSE_train_b, MSE_test_b)
    axes_MSE_2 = np.concatenate(args_MSE, axis=0)

    args_R2 = (R2_train, R2_test, R2_train_c, R2_test_c, R2_train_b, R2_test_b)
    axes_R2_2 = np.concatenate(args_R2, axis=0)

    line_lab_MSE = ['MSE train', 'MSE test', 'MSE_train_c', 'MSE_test_c', 'MSE_train_b', 'MSE_test_b']
    line_lab_R2 = ['R2 train', 'R2 test', 'R2_train_c', 'R2_test_c', 'R2_train_b', 'R2_test_b']

    fig_errors = plt.figure()
    create_plots.make_multi_line_plot(axes_1, axes_MSE_2[0::2], line_lab_MSE[0::2], fig_errors, 221)
    create_plots.make_multi_line_plot(axes_1, axes_R2_2[0::2], line_lab_R2[0::2], fig_errors, 223)
    create_plots.make_multi_line_plot(axes_1, axes_MSE_2[1::2], line_lab_MSE[1::2], fig_errors, 222)
    create_plots.make_multi_line_plot(axes_1, axes_R2_2[1::2], line_lab_R2[1::2], fig_errors, 224)

    plt.show()
