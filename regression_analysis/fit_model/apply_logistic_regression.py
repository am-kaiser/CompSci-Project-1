"""Script to apply different logistic regression methods with resampling and support vector machine algorithms to data."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regression_analysis.fit_model import logistic_regression


def plot_heatmap_conf_matrix(reg_type, l2_lambda, learn_rate, num_min_batch, epoch, test_ratio, k_fold):
    """
    Plot confusion matrices
    :param reg_type: logistic regression or support vector machine
    :param l2_lambda: L2 regularization parameter
    :param learn_rate: learn rate for stochastic gradient descent
    :param num_min_batch: number of mini batches for stochastic gradient descent
    :param epoch: number of epochs for stochastic gradient descent
    :param test_ratio: ratio of data used as a test dataset
    :param k_fold: number of folds to be used with cross-validation
    """
    # Load data
    input_data = logistic_regression.load_data()
    X_obs = logistic_regression.normalise_data(logistic_regression.design_matrix(input_data))
    y_in = input_data.diagnosis.values

    # Create regression object
    logistic_reg = logistic_regression.LogisticRegression(X_obs, y_in)

    # Create empty plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout()
    ax1.title.set_text('Train Data')
    ax2.title.set_text('Test Data')

    # Plot confusion matrix for given input
    if reg_type == "logistic_sgd":
        logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="logistic_sgd", lmbda=l2_lambda, num_epoch=epoch,
                                               learn_rate=learn_rate, num_min_batch=num_min_batch)

        sns.heatmap(logistic_reg.train_confusion_matrix, annot=True, cmap="mako",
                    yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax1)
        sns.heatmap(logistic_reg.test_confusion_matrix, annot=True, cmap="mako",
                    yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax2)

    elif reg_type == "logistic_scikit":
        logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="logistic_scikit")

        sns.heatmap(logistic_reg.train_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax1)
        sns.heatmap(logistic_reg.test_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax2)

    elif reg_type == "svm":
        logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="svm")

        sns.heatmap(logistic_reg.train_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax1)
        sns.heatmap(logistic_reg.test_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax2)

    elif reg_type == "logistic_sgd_crossvalidation":
        logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="logistic_sgd",
                                                               lmbda=l2_lambda, num_epoch=epoch, learn_rate=learn_rate,
                                                               num_min_batch=num_min_batch)

        sns.heatmap(logistic_reg.train_confusion_matrix, annot=True, cmap="mako",
                    yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax1)
        sns.heatmap(logistic_reg.test_confusion_matrix, annot=True, cmap="mako",
                    yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax2)

    elif reg_type == "logistic_scikit_crossvalidation":
        logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="logistic_scikit")

        sns.heatmap(logistic_reg.train_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax1)
        sns.heatmap(logistic_reg.test_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax2)

    elif reg_type == "svm_crossvalidation":
        logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="svm")

        sns.heatmap(logistic_reg.train_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax1)
        sns.heatmap(logistic_reg.test_confusion_matrix, annot=True, cmap="mako", yticklabels=['is_malignant', 'is_benign'],
                    xticklabels=['predicted_malignent', 'predicted_benign'], ax=ax2)


def plot_accuracy(reg_type, l2_lambda, learn_rate, num_min_batch, epoch, test_ratio, k_fold):
    """
    Plot accuracy for test and train data given Logistic Regression with stochastic gradient descent
    :param reg_type: logistic regression with or without cross-validation
    :param l2_lambda: L2 regularization parameter
    :param learn_rate: learn rate for stochastic gradient descent
    :param num_min_batch: number of mini batches for stochastic gradient descent
    :param epoch: number of epochs for stochastic gradient descent
    :param test_ratio: ratio of data used as a test dataset
    :param k_fold: number of folds to be used with cross-validation
    """
    # Load data
    input_data = logistic_regression.load_data()
    X_obs = logistic_regression.normalise_data(logistic_regression.design_matrix(input_data))
    y_in = input_data.diagnosis.values

    # Create regression object
    logistic_reg = logistic_regression.LogisticRegression(X_obs, y_in)

    # Create empty arrays to store test and train accuracy
    test_accr = np.empty(len(learn_rate))
    train_accr = np.empty(len(learn_rate))
    # Calculate test and train accuracy for different learning rates
    for rate_ind, rate in enumerate(learn_rate):
        if reg_type == "logistic_sgd":
            logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="logistic_sgd", lmbda=l2_lambda, num_epoch=epoch,
                                                   learn_rate=rate, num_min_batch=num_min_batch)
            test_accr[rate_ind] = logistic_reg.test_accuracy
            train_accr[rate_ind] = logistic_reg.train_accuracy

        if reg_type == "logistic_sgd_crossvalidation":
            logistic_reg.apply_logistic_regression_crossvalidation(kfolds=k_fold, reg_method="logistic_sgd",
                                                                   lmbda=l2_lambda, num_epoch=epoch, learn_rate=rate,
                                                                   num_min_batch=num_min_batch)
        test_accr[rate_ind] = logistic_reg.test_accuracy
        train_accr[rate_ind] = logistic_reg.train_accuracy

    # Get Scikit Accuracies
    logistic_reg.apply_logistic_regression(test_ratio=test_ratio, reg_method="logistic_scikit")
    test_accr_scikit = logistic_reg.test_accuracy
    train_accr_scikit = logistic_reg.train_accuracy

    # Make plot of accuracies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(learn_rate, train_accr)
    ax1.axhline(y=train_accr_scikit, color='r', linestyle='-')
    ax2.plot(learn_rate, test_accr)
    ax2.axhline(y=test_accr_scikit, color='r', linestyle='-')
    ax1.title.set_text('Train Data')
    ax2.title.set_text('Test Data')
    ax1.set_xlabel('Learn Rate')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Learn Rate')
    ax2.set_ylabel('Accuracy')
    ax1.set_ylim([0.85, 1.05])
    ax2.set_ylim([0.85, 1.05])
    fig.tight_layout()
