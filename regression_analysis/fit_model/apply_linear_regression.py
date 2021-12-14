"""Script to apply different linear regression methods with resampling to data."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regression_analysis.fit_model import linear_regression
from regression_analysis.utils import franke


def create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate, num_min_batch, epochs):
    """Create array filled with zeros."""
    return np.zeros([len(order), len(num_points), len(noise_var), len(test_ratios), len(ridge_lambda), len(lasso_lambda), len(n_boots),
                     len(k_folds), len(learn_rate), len(num_min_batch), len(epochs)])


def apply_regression(order, num_points, noise_var, test_ratio_array=np.zeros(1), reg_type="ols", ridge_lambda=np.ones(1),
                     lasso_lambda=np.ones(1), n_boots=np.ones(1, dtype=int), k_folds=np.ones(1, dtype=int)):
    """
    Apply specified linear regression method to data with given parameters
    :param order: order of polynomial which will be fitted
    :param num_points: number of points to be used for the simulation
    :param noise_var: noise variance to be used in Franke function
    :param test_ratio_array: size of testing data set
    :param reg_type: fitting method to be used
    :param ridge_lambda: lambda for ridge regression
    :param lasso_lambda: lambda for lasso regression
    :param n_boots: number of bootstraps
    :param k_folds: number of folds to be used in cross-validation
    """
    # applies regression for multiple parameter combos
    train_MSE_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(lasso_lambda),
                              len(n_boots), len(k_folds)])
    test_MSE_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(lasso_lambda),
                             len(n_boots), len(k_folds)])
    train_R2_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(lasso_lambda),
                             len(n_boots), len(k_folds)])
    test_R2_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(lasso_lambda),
                            len(n_boots), len(k_folds)])
    # bias in test set
    test_bias_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(lasso_lambda),
                              len(n_boots), len(k_folds)])
    # variance in test set
    test_var_arr = np.zeros([len(order), len(num_points), len(noise_var), len(test_ratio_array), len(ridge_lambda), len(lasso_lambda),
                             len(n_boots), len(k_folds)])
    # Calculate statistical indicators for given regression type and different resampling methods
    for points_ind, num in enumerate(num_points):
        for noise_ind, var in enumerate(noise_var):
            # Create data from Franke function
            xx1, xx2, y = franke.create_data(num_points=num, noise_variance=var)

            linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)

            for order_ind, ordr in enumerate(order):
                for ratio_ind, test_ratio in enumerate(test_ratio_array):
                    if reg_type == "ols":
                        linear_reg.apply_leastsquares(order=ordr, test_ratio=test_ratio, reg_method="ols")
                        train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0] = linear_reg.trainMSE
                        test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0] = linear_reg.testMSE
                        train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0] = linear_reg.trainR2
                        test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0] = linear_reg.testR2
                        test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0] = linear_reg.testbias
                        test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0] = linear_reg.testvar

                    elif reg_type == "ols_bootstrap":
                        for boot_ind, n_boot in enumerate(n_boots):
                            linear_reg.apply_leastsquares_bootstrap(order=ordr, test_ratio=test_ratio,
                                                                    n_boots=n_boot, reg_method="ols")
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0] = linear_reg.testvar

                    elif reg_type == "ols_crossvalidation":
                        # note test_ratio_array is of length one for crossvalidation. we don't need test ratio
                        for fold_ind, k_fold in enumerate(k_folds):
                            linear_reg.apply_leastsquares_crossvalidation(order=ordr, kfolds=k_fold, reg_method="ols")
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind] = linear_reg.testvar

                    elif reg_type == "ridge":
                        for ridge_lam_ind, ridge_lam in enumerate(ridge_lambda):
                            linear_reg.apply_leastsquares(order=ordr, test_ratio=test_ratio, reg_method="ridge",
                                                          lmbda=ridge_lam)
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0] = linear_reg.testvar

                    elif reg_type == "ridge_bootstrap":
                        for ridge_lam_ind, ridge_lam in enumerate(ridge_lambda):
                            for boot_ind, n_boot in enumerate(n_boots):
                                linear_reg.apply_leastsquares_bootstrap(order=ordr, test_ratio=test_ratio,
                                                                        n_boots=n_boot, reg_method="ridge",
                                                                        lmbda=ridge_lam)
                                train_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0] = linear_reg.trainMSE
                                test_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0] = linear_reg.testMSE
                                train_R2_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0] = linear_reg.trainR2
                                test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0] = linear_reg.testR2
                                test_bias_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0] = linear_reg.testbias
                                test_var_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0] = linear_reg.testvar

                    elif reg_type == "ridge_crossvalidation":
                        for ridge_lam_ind, ridge_lam in enumerate(ridge_lambda):
                            for fold_ind, k_fold in enumerate(k_folds):
                                linear_reg.apply_leastsquares_crossvalidation(order=ordr, kfolds=k_fold, reg_method="ridge",
                                                                              lmbda=ridge_lam)
                                train_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind] = linear_reg.trainMSE
                                test_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind] = linear_reg.testMSE
                                train_R2_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind] = linear_reg.trainR2
                                test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind] = linear_reg.testR2
                                test_bias_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind] = linear_reg.testbias
                                test_var_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind] = linear_reg.testvar

                    elif reg_type == "lasso":
                        for lasso_lam_ind, lasso_lam in enumerate(lasso_lambda):
                            linear_reg.apply_leastsquares(order=ordr, test_ratio=test_ratio, reg_method="scikit_lasso",
                                                          lmbda=lasso_lam)
                            train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, 0] = linear_reg.trainMSE
                            test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, 0] = linear_reg.testMSE
                            train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, 0] = linear_reg.trainR2
                            test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, 0] = linear_reg.testR2
                            test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, 0] = linear_reg.testbias
                            test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, 0] = linear_reg.testvar

                    elif reg_type == "lasso_bootstrap":
                        for lasso_lam_ind, lasso_lam in enumerate(lasso_lambda):
                            for boot_ind, n_boot in enumerate(n_boots):
                                linear_reg.apply_leastsquares_bootstrap(order=ordr, test_ratio=test_ratio,
                                                                        n_boots=n_boot, reg_method="scikit_lasso",
                                                                        lmbda=lasso_lam)
                                train_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, boot_ind, 0] = linear_reg.trainMSE
                                test_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, boot_ind, 0] = linear_reg.testMSE
                                train_R2_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, boot_ind, 0] = linear_reg.trainR2
                                test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, boot_ind, 0] = linear_reg.testR2
                                test_bias_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, boot_ind, 0] = linear_reg.testbias
                                test_var_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, boot_ind, 0] = linear_reg.testvar

                    elif reg_type == "lasso_crossvalidation":
                        for lasso_lam_ind, lasso_lam in enumerate(lasso_lambda):
                            for fold_ind, k_fold in enumerate(k_folds):
                                linear_reg.apply_leastsquares_crossvalidation(order=ordr, kfolds=k_fold, reg_method="scikit_lasso",
                                                                              lmbda=lasso_lam)
                                train_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, fold_ind] = linear_reg.trainMSE
                                test_MSE_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, fold_ind] = linear_reg.testMSE
                                train_R2_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, fold_ind] = linear_reg.trainR2
                                test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, fold_ind] = linear_reg.testR2
                                test_bias_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, fold_ind] = linear_reg.testbias
                                test_var_arr[
                                    order_ind, points_ind, noise_ind, ratio_ind, 0, lasso_lam_ind, 0, fold_ind] = linear_reg.testvar

    return train_MSE_arr, test_MSE_arr, train_R2_arr, test_R2_arr, test_bias_arr, test_var_arr


def apply_regression_sgd(order, num_points, noise_var, test_ratios=np.zeros(1), reg_type="ols", ridge_lambda=np.ones(1),
                         lasso_lambda=np.ones(1), n_boots=np.ones(1, dtype=int), k_folds=np.ones(1, dtype=int), learn_rate=np.ones(1),
                         num_min_batch=np.ones(1), epochs=np.ones(1)):
    """
    Apply specified linear regression method with stochastic gradient descent to data with given parameters
    :param order: order of polynomial which will be fitted
    :param num_points: number of points to be used for the simulation
    :param noise_var: noise variance to be used in Franke function
    :param test_ratios: size of testing data set
    :param reg_type: fitting method to be used
    :param ridge_lambda: lambda for ridge regression
    :param lasso_lambda: lambda for lasso regression
    :param n_boots: number of bootstraps
    :param k_folds: number of folds to be used in cross-validation
    :param learn_rate: learn rate for stochastic gradient descent
    :param num_min_batch: number of mini batches for stochastic gradient descent
    :param epochs: number of epochs for stochastic gradient descent
    """
    # applies regression for multiple parameter combos
    train_MSE_arr = create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate,
                             num_min_batch, epochs)
    test_MSE_arr = create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate,
                            num_min_batch, epochs)
    train_R2_arr = create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate,
                            num_min_batch, epochs)
    test_R2_arr = create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate,
                           num_min_batch, epochs)
    # bias in test set
    test_bias_arr = create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate,
                             num_min_batch, epochs)
    # variance in test set
    test_var_arr = create_0(order, num_points, noise_var, test_ratios, ridge_lambda, lasso_lambda, n_boots, k_folds, learn_rate,
                            num_min_batch, epochs)

    # Calculate statistical indicators for given regression type and different resampling methods
    for points_ind, num in enumerate(num_points):
        for noise_ind, var in enumerate(noise_var):
            # Create data from Franke function
            xx1, xx2, y = franke.create_data(num_points=num, noise_variance=var)
            # Create regression object
            linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)

            for order_ind, ordr in enumerate(order):
                for ratio_ind, test_ratio in enumerate(test_ratios):

                    if reg_type == "ols_sgd":
                        for epoch_ind, epoch in enumerate(epochs):
                            for learn_ind, learn_rat in enumerate(learn_rate):
                                for batch_ind, num_batch in enumerate(num_min_batch):
                                    linear_reg.apply_leastsquares(order=ordr, test_ratio=test_ratio, reg_method="ols", num_epoch=epoch,
                                                                  learn_rate=learn_rat, num_min_batch=num_batch)
                                    train_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainMSE
                                    test_MSE_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testMSE
                                    train_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainR2
                                    test_R2_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testR2
                                    test_bias_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testbias
                                    test_var_arr[order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testvar

                    elif reg_type == "ols_bootstrap_sgd":
                        for boot_ind, n_boot in enumerate(n_boots):
                            for epoch_ind, epoch in enumerate(epochs):
                                for learn_ind, learn_rat in enumerate(learn_rate):
                                    for batch_ind, num_batch in enumerate(num_min_batch):
                                        linear_reg.apply_leastsquares_bootstrap(order=ordr, test_ratio=test_ratio, n_boots=n_boot,
                                                                                reg_method="ols", num_epoch=epoch, learn_rate=learn_rat,
                                                                                num_min_batch=num_batch)
                                        train_MSE_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainMSE
                                        test_MSE_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testMSE
                                        train_R2_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainR2
                                        test_R2_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testR2
                                        test_bias_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testbias
                                        test_var_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testvar

                    elif reg_type == "ols_crossvalidation_sgd":
                        # note test_ratio is of length one for crossvalidation. we don't need test ratio
                        for fold_ind, k_fold in enumerate(k_folds):
                            for epoch_ind, epoch in enumerate(epochs):
                                for learn_ind, learn_rat in enumerate(learn_rate):
                                    for batch_ind, num_batch in enumerate(num_min_batch):
                                        linear_reg.apply_leastsquares_crossvalidation(order=ordr, kfolds=k_fold, reg_method="ols",
                                                                                      num_epoch=epoch, learn_rate=learn_rat,
                                                                                      num_min_batch=num_batch)
                                        train_MSE_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.trainMSE
                                        test_MSE_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testMSE
                                        train_R2_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.trainR2
                                        test_R2_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testR2
                                        test_bias_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testbias
                                        test_var_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, 0, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testvar

                    elif reg_type == "ridge_sgd":
                        for ridge_lam_ind, ridge_lam in enumerate(ridge_lambda):
                            for epoch_ind, epoch in enumerate(epochs):
                                for learn_ind, learn_rat in enumerate(learn_rate):
                                    for batch_ind, num_batch in enumerate(num_min_batch):
                                        linear_reg.apply_leastsquares(order=ordr, test_ratio=test_ratio, reg_method="ridge",
                                                                      lmbda=ridge_lam, num_epoch=epoch, learn_rate=learn_rat,
                                                                      num_min_batch=num_batch)
                                        train_MSE_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainMSE
                                        test_MSE_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testMSE
                                        train_R2_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainR2
                                        test_R2_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testR2
                                        test_bias_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testbias
                                        test_var_arr[
                                            order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testvar

                    elif reg_type == "ridge_bootstrap_sgd":
                        for ridge_lam_ind, ridge_lam in enumerate(ridge_lambda):
                            for boot_ind, n_boot in enumerate(n_boots):
                                for epoch_ind, epoch in enumerate(epochs):
                                    for learn_ind, learn_rat in enumerate(learn_rate):
                                        for batch_ind, num_batch in enumerate(num_min_batch):
                                            linear_reg.apply_leastsquares_bootstrap(order=ordr, test_ratio=test_ratio, n_boots=n_boot,
                                                                                    reg_method="ridge", lmbda=ridge_lam, num_epoch=epoch,
                                                                                    learn_rate=learn_rat, num_min_batch=num_batch)
                                            train_MSE_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainMSE
                                            test_MSE_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testMSE
                                            train_R2_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.trainR2
                                            test_R2_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testR2
                                            test_bias_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testbias
                                            test_var_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, boot_ind, 0, learn_ind, batch_ind, epoch_ind] = linear_reg.testvar

                    elif reg_type == "ridge_crossvalidation_sgd":
                        for ridge_lam_ind, ridge_lam in enumerate(ridge_lambda):
                            for fold_ind, k_fold in enumerate(k_folds):
                                for epoch_ind, epoch in enumerate(epochs):
                                    for learn_ind, learn_rat in enumerate(learn_rate):
                                        for batch_ind, num_batch in enumerate(num_min_batch):
                                            linear_reg.apply_leastsquares_crossvalidation(order=ordr, kfolds=k_fold, reg_method="ridge",
                                                                                          lmbda=ridge_lam, num_epoch=epoch,
                                                                                          learn_rate=learn_rat, num_min_batch=num_batch)
                                            train_MSE_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.trainMSE
                                            test_MSE_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testMSE
                                            train_R2_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.trainR2
                                            test_R2_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testR2
                                            test_bias_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testbias
                                            test_var_arr[
                                                order_ind, points_ind, noise_ind, ratio_ind, ridge_lam_ind, 0, 0, fold_ind, learn_ind, batch_ind, epoch_ind] = linear_reg.testvar

    return train_MSE_arr, test_MSE_arr, train_R2_arr, test_R2_arr, test_bias_arr, test_var_arr


def get_data_path():
    """
    Get the directory from which the script is executed to load the data correctly. This is especially important for
    the execution in a jupyter notebook.
    """
    current_path = os.getcwd()
    current_directory = current_path[current_path.rindex(os.sep) + 1:]
    if current_directory == 'examples':
        data_path = 'data_linear_regression/'
    elif current_directory == 'regression_analysis':
        data_path = 'examples/data_linear_regression/'
    elif current_directory == 'CompSci-Project-1':
        data_path = 'regression_analysis/examples/data_linear_regression/'
    else:
        raise Exception('This script is not in the correct directory.')
    return data_path


def get_data_path_sgd():
    """
    Get the directory from which the scripts is executed to load the data correctly. This is especially important for
    the execution in a jupyter notebook.
    """
    current_path = os.getcwd()
    current_directory = current_path[current_path.rindex(os.sep) + 1:]
    if current_directory == 'examples':
        data_path = 'data_linear_regression_sgd/'
    elif current_directory == 'regression_analysis':
        data_path = 'examples/data_linear_regression_sgd/'
    elif current_directory == 'CompSci-Project-1':
        data_path = 'regression_analysis/examples/data_linear_regression_sgd/'
    else:
        raise Exception('This script is not in the correct directory.')
    return data_path


def get_data_statistic(data_path, statistic, method):
    """Load file with given statistical indicator and method."""
    file_name = statistic.replace(' ', '_') + method + '.npy'
    return np.load(data_path + file_name)


def plot_stat(ratio=0.1, num=100, stat="test MSE", method="ols", n_boot=1000, k_fold=1000, ridge_lmb=122.0, lasso_lmb=112.2):
    """
    Create heatmap for given statistical indicator and sampling method
    :param ratio: ratio of the dataset to be used for testing
    :param num: length of dataset
    :param stat: statistical indicator
    :param method: resampling method
    :param n_boot: number of times bootstrap is performed if method=*_bootstrap
    :param k_fold: number of folds for cross-validation if method=*_crossvalidation
    :param ridge_lmb: lambda for ridge regression
    :param lasso_lmb: lambda for lasso regression
    """
    # Path to example data
    data_path = get_data_path()


    # Load data
    order = np.load(data_path + "order.npy")
    num_points = np.load(data_path + "num_points.npy")
    noise_var = np.load(data_path + "noise_var.npy")
    test_ratio = np.load(data_path + "test_ratio.npy")
    ridge_lambda = np.load(data_path + "ridge_lambda.npy")
    k_folds = np.load(data_path + "k_folds.npy")
    n_boots = np.load(data_path + "n_boots.npy")
    lasso_lambda = np.load(data_path + "lasso_lambda.npy")
    learn_rates = np.load(data_path + "learn_rates.npy")
    num_mini_batches = np.load(data_path + "num_min_batches.npy")
    epochs = np.load(data_path + "epochs.npy")

    # Load data for statistical indicator
    data = get_data_statistic(data_path, stat, method)

    n_ind = 0
    for i in range(len(num_points)):
        if num == num_points[i]:
            n_ind = i
    r_ind = 0
    for i in range(len(test_ratio)):
        if ratio == test_ratio[i]:
            r_ind = i
    rlambda_ind = 0
    for i in range(len(ridge_lambda)):
        if ridge_lmb == ridge_lambda[i]:
            rlambda_ind = i
    llambda_ind = 0
    for i in range(len(lasso_lambda)):
        if lasso_lmb == lasso_lambda[i]:
            llambda_ind = i
    nb_ind = 0
    for i in range(len(n_boots)):
        if n_boot == n_boots[i]:
            nb_ind = i
    cv_ind = 0
    for i in range(len(k_folds)):
        if k_fold == k_folds[i]:
            cv_ind = i

    if "crossvalidation" in method:
        r_ind = 0
    else:
        cv_ind = 0
    if "bootstrap" not in method:
        nb_ind = 0
    if "ridge" not in method:
        rlambda_ind = 0
    if "lasso" not in method:
        llambda_ind = 0


    # Select subset of data for given ratio, lambda, number of bootstraps and/or folds for cross-validation and plot heatmap
    data_sub = data[:, n_ind, :, r_ind, rlambda_ind, llambda_ind, nb_ind, cv_ind]
    

    sns.heatmap(data_sub, annot=True, cmap="mako", vmax=np.amax(data_sub), vmin=np.amin(data_sub), xticklabels=noise_var,
                yticklabels=order)
    plt.ylabel('Polynomial Order')
    plt.xlabel('Noise Variance')


def plot_stat_sgd(ratio=0.1, num=100, stat="test MSE", method="ols", n_boot=1000, k_fold=1000, ridge_lmb=122.0, lasso_lmb=112.2, learn_rate=0.1,
              batch=5, epoch=50):
    """
    Create heatmap for given statistical indicator and sampling method
    :param ratio: ratio of the dataset to be used for testing
    :param num: length of dataset
    :param stat: statistical indicator
    :param method: resampling method
    :param n_boot: number of times bootstrap is performed if method=*_bootstrap
    :param k_fold: number of folds for cross-validation if method=*_crossvalidation
    :param ridge_lmb: lambda for ridge regression
    :param lasso_lmb: lambda for lasso regression
    :param epoch: number of epochs for stochastic gradient descent
    :param learn_rate: learn rate for stochastic gradient descent
    :param batch: number of mini batches for stochastic gradient descent
    """
    # Path to example data
    data_path = get_data_path_sgd()

    # Load data
    order = np.load(data_path + "order.npy")
    num_points = np.load(data_path + "num_points.npy")
    noise_var = np.load(data_path + "noise_var.npy")
    test_ratio = np.load(data_path + "test_ratio.npy")
    ridge_lambda = np.load(data_path + "ridge_lambda.npy")
    k_folds = np.load(data_path + "k_folds.npy")
    n_boots = np.load(data_path + "n_boots.npy")
    lasso_lambda = np.load(data_path + "lasso_lambda.npy")
    learn_rates = np.load(data_path + "learn_rates.npy")
    num_mini_batches = np.load(data_path + "num_min_batches.npy")
    epochs = np.load(data_path + "epochs.npy")

    # Load data for statistical indicator
    data = get_data_statistic(data_path, stat, method)

    n_ind = 0
    for i in range(len(num_points)):
        if num == num_points[i]:
            n_ind = i
    r_ind = 0
    for i in range(len(test_ratio)):
        if ratio == test_ratio[i]:
            r_ind = i
    rlambda_ind = 0
    for i in range(len(ridge_lambda)):
        if ridge_lmb == ridge_lambda[i]:
            rlambda_ind = i
    llambda_ind = 0
    for i in range(len(lasso_lambda)):
        if lasso_lmb == lasso_lambda[i]:
            llambda_ind = i
    nb_ind = 0
    for i in range(len(n_boots)):
        if n_boot == n_boots[i]:
            nb_ind = i
    cv_ind = 0
    for i in range(len(k_folds)):
        if k_fold == k_folds[i]:
            cv_ind = i
    for i in range(len(learn_rates)):
        if learn_rate == learn_rates[i]:
            lr_ind = i
    for i in range(len(num_mini_batches)):
        if batch == num_mini_batches[i]:
            b_ind = i
    for i in range(len(epochs)):
        if epoch == epochs[i]:
            e_ind = i

    if "crossvalidation" in method:
        r_ind = 0
    else:
        cv_ind = 0
    if "bootstrap" not in method:
        nb_ind = 0
    if "ridge" not in method:
        rlambda_ind = 0
    if "lasso" not in method:
        llambda_ind = 0
    if "sgd" not in method:
        lr_ind = 0
        b_ind = 0
        e_ind = 0

    # Select subset of data for given ratio, lambda, number of bootstraps and/or folds for cross-validation and plot heatmap
    data_sub = data[:, n_ind, :, r_ind, rlambda_ind, llambda_ind, nb_ind, cv_ind, lr_ind, b_ind, e_ind]

    sns.heatmap(data_sub, annot=True, cmap="mako", vmax=np.amax(data_sub), vmin=np.amin(data_sub), xticklabels=noise_var,
                yticklabels=order)
    plt.ylabel('Polynomial Order')
    plt.xlabel('Noise Variance')