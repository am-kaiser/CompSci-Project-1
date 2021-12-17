
import unittest
import numpy as np


from regression_analysis.utils import stochastic_gradient_descent


class TestSGDMethod(unittest.TestCase):
    def test_SGD_method(self):
        n = 300  # num of data points
        x = 2*np.random.randn(n, 1)
        y = 4 + 3*x   # + np.random.randn(n,1)
        X = np.zeros((n, 2))

        X[:, 0] = 1
        X[:, 1] = x.flatten()
        theta = np.random.randn(2, 1)
        gradient = stochastic_gradient_descent.gradient_RR_OLS
        test = stochastic_gradient_descent.stochastic_gradient_descent_method(
            gradient, y, X, theta, 100, 0.1, 1, 0.0)
        a, b = (test[0][0], test[1][0])
        self.assertAlmostEqual(a, 4)
        self.assertAlmostEqual(b, 3)


if __name__ == '__main__':
    unittest.main()
