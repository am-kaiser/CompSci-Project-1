import numpy as np 
from sklearn.utils import resample
from optimization import learning_schedule, skl_minmaxscaler
from optimization import SGD, SGDmomentum, RMSprop, ADAM, LG, LG_ADAM
import matplotlib.pyplot as plt
import pandas as pd

# get data from sklearn dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split 
cancer = load_breast_cancer()


def test_SGD():
    # Stochastic Gradient Descent Test using vanilla data

    n = 300  # num of data points
    x = 2*np.random.randn(n, 1)
    y = 4+3*x + np.random.randn(n, 1)
    X = np.zeros((n, 2))

    X[:, 0] = 1
    X[:, 1] = x.flatten()

    theta = np.random.randn(2, 1)

    n_epochs = 10000
    M = 100       # size of each minibatch
    m = int(n/M)  # num of minibatches

    t0, t1 = 5, 50

    # initialization
    mt = np.zeros((2, 1))
    st = np.zeros((2, 1))
    t = 0

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            eta = learning_schedule(epoch*m + i, t0, t1)
            t = t + 1
            # theta = SGDmomentum(xi, yi, theta, eta)
            # theta = SGD(xi, yi, theta, eta)
            # theta, st = RMSprop(xi, yi, theta, st, eta)
            theta, mt, st = ADAM(xi, yi, theta, t, mt, st, eta)


# Preprocessing data: loading, splitting, scaling, checking if scaled properly
np.random.seed(2021)
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.20, random_state=0)
X_train = skl_minmaxscaler(X_train)
X_test = skl_minmaxscaler(X_test)

intercept_train = np.ones(X_train.shape[0])[:, np.newaxis]
intercept_test = np.ones(X_test.shape[0])[:, np.newaxis]
X_train = np.concatenate((intercept_train, X_train), axis=1)
X_test = np.concatenate((intercept_test, X_test), axis=1)

assert np.max(X_train)-1 < 1e-9
assert np.min(X_train) < 1e-9


def test_SGD_LG():
    # Initialization
    n = len(X_train)        # number of training data
    num_epochs = 10000      # number of epoch
    M = 50                  # size of each minibatch
    m = int(n/M)            # num of minibatches
    # initial learning schedule values
    t0, t1 = 5, 50
    t = 0
    # initial random values of the parameter
    theta = np.random.randn(X_train.shape[1], 1)

    for epoch in range(num_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_train[random_index:random_index+M]
            yi = y_train[random_index:random_index+M][:, np.newaxis]
            eta = learning_schedule(epoch*m + i, t0, t1)
            t = t + 1
            theta = LG(xi, yi, theta, 0.001, eta)
    # initialization
    mt = np.zeros((X_train.shape[1], 1))
    st = np.zeros((X_train.shape[1], 1))

    # for epoch in range(num_epochs):
    #     for i in range(m):
    #         random_index = np.random.randint(m)
    #         xi = X_train[random_index:random_index+M]
    #         yi = y_train[random_index:random_index+M]
    #         eta = learning_schedule(epoch*m + i, t0, t1)
    #         t = t + 1
    #         # theta = SGDmomentum(xi, yi, theta, eta)
    #         # theta = SGD(xi, yi, theta, eta)
    #         # theta, st = RMSprop(xi, yi, theta, st, eta)
    #         theta, mt, st = LG_ADAM(xi, yi, theta, t, mt, st, eta)

    # make a prediction
    y_raw = X_test @ theta
    y_pred = []
    # print(theta.shape)
    for i in range(len(y_raw)):
        if y_raw[i:i+1] >= 0:
            y_pred.append(1)
        elif y_raw[i:i+1] < 0:
            y_pred.append(0)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    return accuracy


def test_skl_LG():
    from sklearn.linear_model import LogisticRegression

    # train
    clf_lg = LogisticRegression(random_state=0)
    clf_lg.fit(X_train, y_train)

    # predict
    y_predict_lg = clf_lg.predict(X_test)

    # calculate accuracy
    accuracy = np.mean(y_predict_lg == y_test)
    return accuracy


def test_skl_SVM():
    from sklearn import svm

    # train
    clf_svm = svm.SVC()
    clf_svm.fit(X_train, y_train)

    # predict
    y_predict_svm = clf_svm.predict(X_test)

    # calculate accuracy
    accuracy = np.mean(y_predict_svm == y_test)
    return accuracy


print(f'SG Logistic regression accuracy = {test_SGD_LG()}')
print(f'SKL Logistic Regression accuracy = {test_skl_LG()}')
print(f'SKL SVM accuracy = {test_skl_SVM()}')
