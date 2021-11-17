import numpy as np
from numpy import random as npr


def bootstrap(x, y, sample_ratio):
    n = np.size(y)
    sample_size = int(np.floor(sample_ratio*n))
    train_ind = np.zeros(sample_size, dtype=int)
    ind = np.arange(0,n)
    test_ind = npr.choice(ind, sample_size)
    train_ind = np.delete(ind, test_ind)

    x_train = x[train_ind]
    x_test = x[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]

    return x_train, x_test, y_train, y_test

def crossvalidation(x, y, kfolds):
    n = np.size(y)
    nk = int(np.floor(n/kfolds)) #number of points per fold

    #shuffling
    ind = np.arange(n)
    npr.shuffle(ind)
    x = x[ind]
    y = y[ind]

    x_test_arr = np.zeros([kfolds, nk, 2])
    y_test_arr = np.zeros([kfolds, nk])

    x_train_arr = np.zeros([kfolds, n-nk, 2])
    y_train_arr = np.zeros([kfolds, n-nk])

    for k in np.arange(kfolds):
        test_ind = np.arange(k*nk,(k+1)*nk)
        train_ind = np.delete(np.arange(n), test_ind)

        x_test_arr[k,:,:] = x[test_ind]
        y_test_arr[k,:] = y[test_ind,0]
        x_train_arr[k,:,:] = x[train_ind]
        y_train_arr[k,:] = y[train_ind,0]

    return x_train_arr, x_test_arr, y_train_arr, y_test_arr

if __name__ == '__main__':
    x = np.linspace(0, 1, 10)
    y = np.linspace(3, 5, 10)
    sample_ratio = 0.31
    print(bootstrap(x,y,sample_ratio))

