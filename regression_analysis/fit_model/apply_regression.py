"""Script to apply different regression methods with resampling to data."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regression_analysis.fit_model import linear_regression
from regression_analysis.utils import franke


def apply_regression(order, num_points, noise_var, test_ratio_array=np.zeros(1), reg_type="ols", ridge_lambda=np.ones(1),
                     n_boots=np.ones(1, dtype=int), k_folds=np.ones(1, dtype=int)):
    # applies regression for multiple parameter combos
    train_MSE_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(n_boots),
                              len(k_folds)])
    test_MSE_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(n_boots),
                             len(k_folds)])
    train_R2_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(n_boots),
                             len(k_folds)])
    test_R2_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(n_boots),
                            len(k_folds)])
    # bias in test set
    test_bias_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(n_boots),
                              len(k_folds)])
    # variance in test set
    test_var_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(n_boots),
                             len(k_folds)])

    # Calculate statistical indicators for given regression type and different resampling methods
    for points_ind in range(len(num_points)):
        for noise_ind in range(len(noise_var)):
            # Create data from Franke function
            xx1, xx2, y = franke.create_data(num_points=num_points[points_ind], noise_variance=noise_var[noise_ind])

            linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)

            for order_ind in range(len(order)):
                for ratio_ind in range(len(test_ratio_array)):
                    if reg_type == "ols":
                        linear_reg.apply_leastsquares(order=order[order_ind], test_ratio=test_ratio_array[ratio_ind])
                        train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0] = linear_reg.trainMSE
                        test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0] = linear_reg.testMSE
                        train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0] = linear_reg.trainR2
                        test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0] = linear_reg.testR2
                        test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0] = linear_reg.testbias
                        test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0] = linear_reg.testvar

                    elif reg_type == "ols_bootstrap":
                        for boot_ind in range(len(n_boots)):
                            linear_reg.apply_leastsquares_bootstrap(order=order[order_ind], test_ratio=test_ratio_array[ratio_ind],
                                                                    n_boots=n_boots[boot_ind])
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, boot_ind, 0] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, boot_ind, 0] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, boot_ind, 0] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, boot_ind, 0] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, boot_ind, 0] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, boot_ind, 0] = linear_reg.testvar

                    elif reg_type == "ols_crossvalidation":
                        # note test_ratio_array is of length one for crossvalidation. we don't need test ratio
                        for fold_ind in range(len(k_folds)):
                            linear_reg.apply_leastsquares_crossvalidation(order=order[order_ind], kfolds=k_folds[fold_ind])
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, fold_ind] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, fold_ind] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, fold_ind] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, fold_ind] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, fold_ind] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, fold_ind] = linear_reg.testvar

                    elif reg_type == "ridge":
                        for ridge_lam_ind in range(len(ridge_lambda)):
                            linear_reg.apply_leastsquares(order=order[order_ind], test_ratio=test_ratio_array[ratio_ind], ridge=True,
                                                          lmbda=ridge_lambda[ridge_lam_ind])
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0] = linear_reg.testvar

                    elif reg_type == "ridge_bootstrap":
                        for ridge_lam_ind in range(len(ridge_lambda)):
                            for boot_ind in range(len(n_boots)):
                                linear_reg.apply_leastsquares_bootstrap(order=order[order_ind], test_ratio=test_ratio_array[ratio_ind],
                                                                        n_boots=n_boots[boot_ind], ridge=True,
                                                                        lmbda=ridge_lambda[ridge_lam_ind])
                                train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, boot_ind, 0] = linear_reg.trainMSE
                                test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, boot_ind, 0] = linear_reg.testMSE
                                train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, boot_ind, 0] = linear_reg.trainR2
                                test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, boot_ind, 0] = linear_reg.testR2
                                test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, boot_ind, 0] = linear_reg.testbias
                                test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, boot_ind, 0] = linear_reg.testvar

                    elif reg_type == "ridge_crossvalidation":
                        for ridge_lam_ind in range(len(ridge_lambda)):
                            for fold_ind in range(len(k_folds)):
                                linear_reg.apply_leastsquares_crossvalidation(order=order[order_ind], kfolds=k_folds[fold_ind], ridge=True,
                                                                              lmbda=ridge_lambda[ridge_lam_ind])
                                train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, fold_ind] = linear_reg.trainMSE
                                test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, fold_ind] = linear_reg.testMSE
                                train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, fold_ind] = linear_reg.trainR2
                                test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, fold_ind] = linear_reg.testR2
                                test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, fold_ind] = linear_reg.testbias
                                test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, fold_ind] = linear_reg.testvar

    return train_MSE_arr, test_MSE_arr, train_R2_arr, test_R2_arr, test_bias_arr, test_var_arr


def get_data_path():
    """
    Get the directory from which the scripts is executed to load the data correctly. This is especially important for
    the execution in a jupyter notebook.
    """
    current_path = os.getcwd()
    current_directory = current_path[current_path.rindex('/') + 1:]
    if current_directory == 'examples':
        data_path = 'data/'
    elif current_directory == 'regression_analysis':
        data_path = 'examples/data/'
    elif current_directory == 'CompSci-Project-1':
        data_path = 'regression_analysis/examples/data/'
    else:
        raise Exception('This script is not in the correct directory.')
    return data_path


def get_data_statistic(data_path, statistic, method):
    """Load file with given statistical indicator and method."""
    file_name = statistic.replace(' ', '_') + method + '.npy'
    return np.load(data_path + file_name)


def plot_stat(ratio=0.1, num=100, stat="test MSE", method="ols", n_boot=1000, k_fold=1000, ridge_lmb=122.0):
    """"
    Create heatmap for given statistical indicator and sampling method.
    :param ratio: ratio of the dataset to be used for testing
    :param num: length of dataset
    :param stat: statistical indicator
    :param method: resampling method
    :param n_boot: number of times bootstrap is performed if method=*_bootstrap
    :param k_fold: number of folds for cross-validation if method=*_crossvalidation
    :param ridge_lmb: lambda for ridge regression
    """
    # Path to example data
    data_path = get_data_path()
    # Load data
    order = np.load(data_path + "order.npy")
    num_points = np.load(data_path + "num_points.npy")
    noise_var = np.load(data_path + "noise_var.npy")
    test_ratio_array = np.load(data_path + "test_ratio_array.npy")
    ridge_lambda = np.load(data_path + "ridge_lambda.npy")
    k_folds = np.load(data_path + "k_folds.npy")
    n_boots = np.load(data_path + "n_boots.npy")

    # Load data for statistical indicator
    data = get_data_statistic(data_path, stat, method)

    n_ind = 0
    for i in range(len(num_points)):
        if num == num_points[i]:
            n_ind = i
    r_ind = 0
    for i in range(len(test_ratio_array)):
        if ratio == test_ratio_array[i]:
            r_ind = i
    lambda_ind = 0
    for i in range(len(ridge_lambda)):
        if ridge_lmb == ridge_lambda[i]:
            lambda_ind = i
    nb_ind = 0
    for i in range(len(n_boots)):
        if n_boot == n_boots[i]:
            nb_ind = i
    cv_ind = 0
    for i in range(len(k_folds)):
        if k_fold == k_folds[i]:
            cv_ind = i

    if method == "ols_crossvalidation" or method == "ridge_crossvalidation":
        r_ind = 0
    if method == "ols" or method == "ols_crossvalidation" or method == "ridge" or method == "ridge_crossvalidation":
        nb_ind = 0
    if method == "ols" or method == "ols_bootstrap" or method == "ridge" or method == "ridge_bootstrap":
        cv_ind = 0
    if method == "ols" or method == "ols_bootstrap" or method == "ols_crossvalidation":
        lambda_ind = 0

    # Select subset of data for given ratio, lambda, number of bootstraps and/or folds for cross-validation and plot
    # heatmap
    data_sub = data[:, n_ind, :, r_ind, lambda_ind, nb_ind, cv_ind]
    sns.heatmap(data_sub, annot=True, cmap="mako", vmax=np.amax(data_sub), vmin=np.amin(data_sub), xticklabels=noise_var,
                yticklabels=order)
    plt.ylabel('Polynomial Order')
    plt.xlabel('Noise Variance')


if __name__ == "__main__":
    # Plot one heatmap as an example
    plot_stat(ratio=0.1, num=100, stat="test MSE", method="ols", n_boot=1000, k_fold=1000, ridge_lmb=122.0)
    plt.show()
