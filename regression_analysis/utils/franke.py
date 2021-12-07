import numpy as np
from numpy import random as npr


def Franke(x1, x2, noise_var=0.0):
    a = 0.75 * np.exp(-((9 * x1 - 2) ** 2) / 4 - ((9 * x2 - 2) ** 2) / 4)
    b = 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49 - (9 * x2 + 1) / 10)
    c = 0.5 * np.exp(-((9 * x1 - 7) ** 2) / 4 - ((9 * x2 - 3) ** 2) / 4)
    d = 0.2 * np.exp(-((9 * x1 - 4) ** 2) - ((9 * x2 - 7) ** 2))
    e = npr.normal(loc=0, scale=noise_var, size=a.shape)
    #note if we don't truncate the gaussian, we won't see the effect of the noise 
    #as all noise will be rescaled to the range 0,1.
    e[np.where(e>1.0)]=1.0
    e[np.where(e<-1.0)]=-1.0
    return a + b + c - d + e


def create_data(num_points, noise_variance):
    x1 = np.linspace(0, 1, num_points)
    x2 = np.linspace(0, 1, num_points)
    xx1, xx2 = np.meshgrid(x1, x2)

    xx1 = xx1.reshape((num_points * num_points), 1)
    xx2 = xx2.reshape((num_points * num_points), 1)
    y = Franke(xx1, xx2, noise_var=noise_variance)

    return xx1, xx2, y

def Franke_and_biascorrection(x1, x2, noise_var=0.0):
    a = 0.75 * np.exp(-((9 * x1 - 2) ** 2) / 4 - ((9 * x2 - 2) ** 2) / 4)
    b = 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49 - (9 * x2 + 1) / 10)
    c = 0.5 * np.exp(-((9 * x1 - 7) ** 2) / 4 - ((9 * x2 - 3) ** 2) / 4)
    d = 0.2 * np.exp(-((9 * x1 - 4) ** 2) - ((9 * x2 - 7) ** 2))
    e = npr.normal(loc=0, scale=noise_var, size=a.shape)
    #note if we don't truncate the gaussian, we won't see the effect of the noise 
    #as all noise will be rescaled to the range 0,1.
    e[np.where(e>1.0)]=1.0
    e[np.where(e<-1.0)]=-1.0
    return a + b + c - d + e, np.var(e)

if __name__ == '__main__':
    print(create_data(100, 0.1))
