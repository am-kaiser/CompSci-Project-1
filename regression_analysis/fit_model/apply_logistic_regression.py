"""Script to apply different logistic regression methods with resampling to data."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regression_analysis.fit_model import logistic_regression


def create_0(num_points, test_ratios, l2_lambda, k_folds, learn_rate, num_min_batch, epochs):
    """Create array filled with zeros."""
    return np.zeros([len(num_points), len(test_ratios), len(l2_lambda), len(k_folds), len(learn_rate), len(num_min_batch), len(epochs)]) #2,2 for confusion matrix


def apply_regression(num_points, test_ratios=np.zeros(1), reg_type="logistic_sgd", l2_lambda=np.ones(1), k_folds=np.ones(1, dtype=int),
                     learn_rate=np.ones(1), num_min_batch=np.ones(1), epochs=np.ones(1)):
    # create empty numpy arrays for train and test accuracy
    train_acc_arr = create_0(num_points, test_ratios, l2_lambda, k_folds, learn_rate, num_min_batch, epochs)
    test_acc_arr = create_0(num_points, test_ratios, l2_lambda, k_folds, learn_rate, num_min_batch, epochs)
    # create empty numpy arrays for train and test confusion matrix
    #train_conf_mat_arr = create_0(num_points, test_ratios, l2_lambda, k_folds, learn_rate, num_min_batch, epochs)
    #test_conf_mat_arr = create_0(num_points, test_ratios, l2_lambda, k_folds, learn_rate, num_min_batch, epochs)
    
    # Calculate statistical indicators for given regression type and different resampling methods
    for points_ind, num in enumerate(num_points):
        # Load data
        input_data = logistic_regression.load_data()
        X_obs = logistic_regression.normalise_data(logistic_regression.design_matrix(input_data))
        y_in = input_data.diagnosis.values
        # Create regression object
        logistic_reg = logistic_regression.LogisticRegression(X_obs, y_in)

        for ratio_ind, test_ratio in enumerate(test_ratios):
            if reg_type == "logistic_sgd":
                for lam_ind, lam in enumerate(l2_lambda):
                    for epoch_ind, epoch in enumerate(epochs):
                        for learn_ind, learn_rat in enumerate(learn_rate):
                            for batch_ind, num_batch in enumerate(num_min_batch):
                                logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="logistic_sgd", lmbda=lam,
                                                                       num_epoch=epoch,
                                                                       learn_rate=learn_rat, num_min_batch=num_batch)
                                train_acc_arr[points_ind, ratio_ind, lam_ind, 0, learn_ind, batch_ind, epoch_ind] = logistic_reg.train_accuracy
                                test_acc_arr[points_ind, ratio_ind, lam_ind, 0, learn_ind, batch_ind, epoch_ind] = logistic_reg.test_accuracy
                                #train_conf_mat_arr[points_ind, ratio_ind, lam_ind, 0, learn_ind, batch_ind, epoch_ind] = logistic_reg.train_confusion_matrix
                                #test_conf_mat_arr[points_ind, ratio_ind, lam_ind, 0, learn_ind, batch_ind, epoch_ind] = logistic_reg.test_confusion_matrix

            if reg_type == "logistic_scikit":
                logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="logistic_scikit")
                train_acc_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.train_accuracy
                test_acc_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.test_accuracy
                #train_conf_mat_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.train_confusion_matrix
                #test_conf_mat_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.test_confusion_matrix

            if reg_type == "svm":
                logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="svm")
                train_acc_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.train_accuracy
                test_acc_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.test_accuracy
                #train_conf_mat_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.train_confusion_matrix
                #test_conf_mat_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.test_confusion_matrix

        for fold_ind, k_fold in enumerate(k_folds):
            if reg_type == "logistic_sgd_crossvalidation":
                for lam_ind, lam in enumerate(l2_lambda):
                    for epoch_ind, epoch in enumerate(epochs):
                        for learn_ind, learn_rat in enumerate(learn_rate):
                            for batch_ind, num_batch in enumerate(num_min_batch):
                                logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="logistic_sgd",
                                                                                       lmbda=lam, num_epoch=epoch, learn_rate=learn_rat,
                                                                                       num_min_batch=num_batch)
                                train_acc_arr[points_ind, 0, lam_ind, fold_ind, learn_ind, batch_ind, epoch_ind] = logistic_reg.train_accuracy
                                test_acc_arr[points_ind, 0, lam_ind, fold_ind, learn_ind, batch_ind, epoch_ind] = logistic_reg.test_accuracy
                                #train_conf_mat_arr[points_ind, 0, lam_ind, fold_ind, learn_ind, batch_ind, epoch_ind] = logistic_reg.train_confusion_matrix
                                #test_conf_mat_arr[points_ind, 0, lam_ind, fold_ind, learn_ind, batch_ind, epoch_ind] = logistic_reg.test_confusion_matrix

            if reg_type == "logistic_scikit_crossvalidation":
                logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="logistic_scikit")
                train_acc_arr[points_ind, 0, 0, fold_ind, 0, 0, 0] = logistic_reg.train_accuracy
                test_acc_arr[points_ind, 0, 0, fold_ind, 0, 0, 0] = logistic_reg.test_accuracy
                #train_conf_mat_arr[points_ind, 0, 0, fold_ind, 0, 0, 0] = logistic_reg.train_confusion_matrix
                #test_conf_mat_arr[points_ind, 0, 0, fold_ind, 0, 0, 0] = logistic_reg.test_confusion_matrix

            if reg_type == "svm_crossvalidation":
                logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="svm")
                train_acc_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.train_accuracy
                test_acc_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.test_accuracy
                #train_conf_mat_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.train_confusion_matrix
                #test_conf_mat_arr[points_ind, ratio_ind, 0, 0, 0, 0, 0] = logistic_reg.test_confusion_matrix

    return train_acc_arr, test_acc_arr, #train_conf_mat_arr, test_conf_mat_arr


def get_data_path():
    """
    Get the directory from which the scripts is executed to load the data correctly. This is especially important for
    the execution in a jupyter notebook.
    """
    current_path = os.getcwd()
    current_directory = current_path[current_path.rindex(os.sep) + 1:]
    if current_directory == 'examples':
        data_path = 'data_logistic_regression/'
    elif current_directory == 'regression_analysis':
        data_path = 'examples/data_logistic_regression/'
    elif current_directory == 'CompSci-Project-1':
        data_path = 'regression_analysis/examples/data_logistic_regression/'
    else:
        raise Exception('This script is not in the correct directory.')
    return data_path


def get_data_statistic(data_path, statistic, method):
    """Load file with given statistical indicator and method."""
    file_name = statistic.replace(' ', '_') + method + '.npy'
    return np.load(data_path + file_name)


def plot_stat(ratio=0.1, num=100, stat="test accuracy", method="logistic_sgd", k_fold=1000, l2_lambda=1, learn_rate=0.1, batch=5, epoch=50):
    """
    Create heatmap for given statistical indicator and sampling method
    :param ratio: ratio of the dataset to be used for testing
    :param num: length of dataset
    :param stat: statistical indicator
    :param method: resampling method
    :param k_fold: number of folds for cross-validation if method=*_crossvalidation
    :param l2_lambda: L2 regularization parameter
    :param epoch: number of epochs for stochastic gradient descent
    :param learn_rate: learn rate for stochastic gradient descent
    :param batch: number of mini batches for stochastic gradient descent
    """
    # Path to example data
    data_path = get_data_path()
    # Load data
    num_points = np.load(data_path + "num_points.npy")
    test_ratio = np.load(data_path + "test_ratio.npy")
    l2_lambdas = np.load(data_path + "l2_lambda.npy")
    k_folds = np.load(data_path + "k_folds.npy")
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
    lambda_ind = 0
    for i in range(len(l2_lambdas)):
        if l2_lambda == l2_lambdas[i]:
            lambda_ind = i
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
    if "sgd" not in method:
        lr_ind = 0
        b_ind = 0
        e_ind = 0
        lambda_ind = 0

    # Select subset of data for given ratio, lambda, number of folds for cross-validation and plot heatmap
    if "crossvalidation" not in method:
        data_sub = data[:, :, lambda_ind, cv_ind, lr_ind, b_ind, e_ind]
        sns.heatmap(data_sub, annot=True, cmap="mako", vmax=np.amax(data_sub), vmin=np.amin(data_sub), xticklabels=test_ratio,
                    yticklabels=num_points)
        plt.ylabel('Number of Points')
        plt.xlabel('Testing Ratio')
    else:
        data_sub = data[:, r_ind, lambda_ind, :, lr_ind, b_ind, e_ind]
        sns.heatmap(data_sub, annot=True, cmap="mako", vmax=np.amax(data_sub), vmin=np.amin(data_sub), xticklabels=k_folds,
                    yticklabels=num_points)
        plt.ylabel('Number of Points')
        plt.xlabel('Number of Folds')


if __name__ == "__main__":
    # Plot one heatmap as an example
    plot_stat(ratio=0.1, num=100, stat="test MSE", method="logistic_sgd", k_fold=1000, l2_lambda=1, learn_rate=0.1, batch=5, epoch=50)
    plt.show()
