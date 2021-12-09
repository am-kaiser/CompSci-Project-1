"""Script to apply different logistic regression methods with resampling to data."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regression_analysis.fit_model import logistic_regression


def plot_heatmap_conf_matrix(reg_type, l2_lambda, learn_rate, num_min_batch, epoch, test_ratio, k_fold):
    # Calculate statistical indicators for given regression type and different resampling methods
    # Load data
    input_data = logistic_regression.load_data()
    X_obs = logistic_regression.normalise_data(logistic_regression.design_matrix(input_data))
    y_in = input_data.diagnosis.values
    # Create regression object
    logistic_reg = logistic_regression.LogisticRegression(X_obs, y_in)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout()
    ax1.title.set_text('Train Data')
    ax2.title.set_text('Test Data')

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
    # Calculate statistical indicators for given regression type and different resampling methods
    # Load data
    input_data = logistic_regression.load_data()
    X_obs = logistic_regression.normalise_data(logistic_regression.design_matrix(input_data))
    y_in = input_data.diagnosis.values
    # Create regression object
    logistic_reg = logistic_regression.LogisticRegression(X_obs, y_in)

    test_accr = np.empty(len(learn_rate))
    train_accr = np.empty(len(learn_rate))

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(learn_rate, train_accr)
    ax2.plot(learn_rate, test_accr)
    ax1.title.set_text('Train Data')
    ax2.title.set_text('Test Data')
    ax1.set_xlabel('Learn Rate')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Learn Rate')
    ax2.set_ylabel('Accuracy')
    ax1.set_ylim([0.8, 1])
    ax2.set_ylim([0.8, 1])
    fig.tight_layout()


if __name__ == "__main__":
    plot_accuracy(reg_type="logistic_sgd", l2_lambda=0, learn_rate=[0, 1, 2, 3], num_min_batch=4, epoch=5, test_ratio=0.1, k_fold=None)
    plt.show()
