import numpy as np
from regression_analysis.fit_model import logistic_regression

'''
source: https://en.wikipedia.org/wiki/Confusion_matrix
             cancer   non-cancer
cancer          6         2
non-cancer      1         3
'''


def test_confusion_mat():
    actual = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    predicted = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    df = logistic_regression.make_confusion_matrix(actual, predicted)
    print(df)


test_confusion_mat()
