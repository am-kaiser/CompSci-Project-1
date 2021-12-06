'''
OLS grad => (2/len(yi))*xi.T @((xi @ theta) - yi)
Ridge grad => (2/len(yi))*xi.T @ (xi @ (theta)-yi)+2*lmbda*theta
'''
import numpy as np 


def SGD(xi, yi, theta, eta, lmbda=0.001):
    # grad = (2/len(yi))*xi.T @((xi @ theta) - yi)
    grad = (2/len(yi))*xi.T @ (xi @ (theta)-yi)+2*lmbda*theta
    theta = theta - eta*grad
    return theta


def SGDmomentum(xi, yi, theta, eta, momentum=0.9, lmbda=0.001):
    vt = np.zeros((xi.shape[1], yi.shape[1]))
    # gt = 2*xi.T @((xi @ theta) - yi)
    gt = (2/len(yi))*xi.T @ (xi @ (theta)-yi)+2*lmbda*theta
    vt = momentum*vt + eta*gt
    theta = theta - vt
    return theta


def RMSprop(xi, yi, theta, st, eta=1e-3, beta=0.9, eps=1e-8, lmbda=0.001):
    # gt = 2*xi.T @((xi @ theta) - yi)
    gt = (2/len(yi))*xi.T @ (xi @ (theta)-yi)+2*lmbda*theta
    st = beta*st + (1-beta)*gt*gt
    theta = theta - eta*gt/(np.sqrt(st + eps))
    return theta, st


def ADAM(xi, yi, theta, t, mt, st, eta, beta1=0.90, beta2=0.99, eps=1e-7, lmbda=0.001):
    # gt = 2*xi.T @((xi @ theta) - yi)
    gt = (2/len(yi))*xi.T @ (xi @ (theta)-yi)+2*lmbda*theta
    mt = beta1*mt + (1-beta1)*gt
    st = beta2*st + (1-beta2)*gt*gt
    mt = mt/(1-beta1**t)
    st = st/(1-beta2**t)
    theta = theta - eta*mt/(np.sqrt(st)+eps)
    return theta, mt, st


def sigmoid(beta, X):
    assert beta.shape[0] == X.shape[1]
    t = X @ beta
    p = np.exp(t)/(1+np.exp(t))
    return p


def LG(xi, yi, theta, lmbda=0, eta=1e-3):
    p = sigmoid(theta, xi)
    b = -(1/len(yi))*xi.T @ (yi - p) - lmbda*theta
    theta = theta - eta*b
    return theta

def LG_ADAM(xi, yi, theta, t, mt, st, eta, beta1=0.90, beta2=0.99, eps=1e-7, lmbda=0.001):
    # gt = 2*xi.T @((xi @ theta) - yi)
    p = sigmoid(theta, xi)
    gt = (2/len(yi))*xi.T @ (p - yi)+2*lmbda*theta
    print(gt.shape)
    mt = beta1*mt + (1-beta1)*gt
    st = beta2*st + (1-beta2)*gt*gt
    mt = mt/(1-beta1**t)
    st = st/(1-beta2**t)
    theta = theta - eta*mt/(np.sqrt(st)+eps)
    return theta, mt, st

def learning_schedule(t, t0, t1):
    return t0/(t + t1)


def skl_minmaxscaler(X):
    from sklearn.preprocessing import MinMaxScaler
    X_scaled = MinMaxScaler().fit_transform(X)
    return X_scaled

