import numpy as np
from numpy import random as npr


def bootstrap(x, y, sample_ratio):
    """
    generates bootstrap samples
    """
    n = np.size(y)
    sample_size = int(np.ceil((1 - sample_ratio) * n))
    train_ind = np.zeros(sample_size, dtype=int)
    ind = np.arange(0, n)
    train_ind = npr.choice(ind, sample_size)
    # test_ind = npr.choice(np.delete(ind, train_ind), n-sample_size)
    test_ind = np.delete(ind, train_ind)

    x_train = x[train_ind]
    x_test = x[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]

    return x_train, x_test, y_train, y_test


def crossvalidation(x, y, kfolds):
    """
    generates cross validation samples
    """
    n = np.size(y)
    nk = int(np.floor(n / kfolds))  # number of points per fold
    num_col = x.shape[1]

    # shuffling
    ind = np.arange(n)
    npr.shuffle(ind)
    x = x[ind]
    y = y[ind]

    x_test_arr = np.zeros([kfolds, nk, num_col])
    y_test_arr = np.zeros([kfolds, nk])

    x_train_arr = np.zeros([kfolds, n - nk, num_col])
    y_train_arr = np.zeros([kfolds, n - nk])

    for k in np.arange(kfolds):
        test_ind = np.arange(k * nk, (k + 1) * nk)
        train_ind = np.delete(np.arange(n), test_ind)

        x_test_arr[k, :, :] = x[test_ind]
        y_test_arr[k, :] = y[test_ind, 0]
        x_train_arr[k, :, :] = x[train_ind]
        y_train_arr[k, :] = y[train_ind, 0]

    return x_train_arr, x_test_arr, y_train_arr, y_test_arr


if __name__ == '__main__':
    x = np.linspace(1, 10, 10)
    y = np.linspace(4, 13, 10)
    sample_ratio = 0.31
    print(bootstrap(x, y, sample_ratio))
