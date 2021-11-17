"""Define classes/ functions which built the fundament for linear regression analysis."""
import numpy as np


# Define class which contains design matrix with 2D input
class Design_Matrix_2D:
    def __init__(self, x1, x2, f, order):
        self.x1 = x1
        self.x2 = x2
        self.f = f
        self.order = order

        # Flatten values
        self.x1_flat = x1.flatten()
        self.x2_flat = x2.flatten()
        self.f_flat = f.flatten()

    def make_ploynomial_power(self):
        """Create array with the powers of the polynomial which will be fitted."""
        order_range = range(self.order+1)
        order_options = np.array(np.meshgrid(order_range, order_range)).T.reshape(-1, 2)
        row_sum = np.sum(order_options, axis=1) <= self.order
        return order_options[row_sum, :]

    def make_design_matrix(self):
        """Define design matrix D for given input. Order is the order of the polynomial which will be
        fitted. Note first column is vectors with only 1 as values"""
        # Get array with powers for polynomials
        powers = self.make_ploynomial_power()
        # Create empty design matrix
        D = np.empty([len(self.x1_flat), powers.shape[0]])
        # The columns of the design matrix are x1^i*x2^j where i,j in range(order+1)
        for column, power in enumerate(powers):
            D[:, column] = self.x1_flat**power[0]*self.x2_flat**power[1]

        return D

    def calculate_matrix_inverse(self, A):
        """Calculate the inverse of a matrix."""
        # Decide which algorithm will be used to calculate the inverse
        # Note: The condition number of a matrix is a measure of how close the matrix is to being singular. A large
        # condition number means that the matrix is close to being singular.
        if np.linalg.cond(A) < 1/np.finfo(float).eps:
            A_inverse = np.linalg.inv(A)
        # Else use singular value decomposition
        else:
            U, S, V_t = np.linalg.svd(A)
            S_inverse = np.linalg.inv(np.diag(S))
            A_inverse = np.dot(V_t.T, np.dot(S_inverse, U.T))
        return A_inverse

    def design_matrix_product_inverse(self):
        """To calculate any parameter vectors/beta for ordinary least squares inverse of A=(D.T*D) has to be
        calculated. D is  the design matrix. """
        # Get design matrix
        D = self.make_design_matrix()
        A = np.dot(D.T, D)
        # Get inverse of A
        A_inverse = self.calculate_matrix_inverse(A)

        return A_inverse


# Class which contains functionality to judge fit of the model
class Error_Measures:
    def __init__(self, observed_values, model_values):
        self.observed_values = observed_values.flatten()
        self.model_values = model_values.flatten()

    def mean_squared_error(self):
        """Calculate mean squared error of fitted model"""
        MSE = (1/len(self.observed_values))*np.sum((self.observed_values - self.model_values)**2)
        return MSE

    def r2_score(self):
        """Calculate R2 score"""
        oben = len(self.observed_values)*self.mean_squared_error()
        unten = np.sum((self.observed_values - np.mean(self.model_values))**2)
        R2 = 1 - (oben/unten)
        return R2
