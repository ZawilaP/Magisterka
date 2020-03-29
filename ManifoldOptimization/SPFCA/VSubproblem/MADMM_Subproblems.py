from ManifoldOptimization.Utils.matrix_operations import numpy_to_pandas, pandas_to_numpy, elementwise_multiplication, get_matrix_sign, get_matrix_transpose, get_matrix_inverse, get_matrix_multiplication
import numpy as np
import pandas as pd

class WSubproblem():

    def __init__(self, lambda_1, V_matrix, Z_matrix, W_matrix, rho):
        self.lambda_constant = lambda_1
        self.V = V_matrix
        self.Z = Z_matrix
        self.W = W_matrix
        self.rho = rho
        self.Y = self.V + self.Z
        self.W_k = self.compute_new_W_k()

    def __call__(self, *args, **kwargs):
        return self.W_k

    def compute_new_W_k(self):
        '''
        Applies elementwise soft-thresholding and elementwise multiplication by signs of matrix
        Using conversion from numpy to pandas, for computational sake, as applymap is more optimized than
        numpy's vectorize, which will be beneficial for huge matrices
        :return: new value of W
        '''
        rho = self.rho
        lambda_constant = self.lambda_constant

        def soft_threshold_for_matrix(element):
            return max([0, element - (lambda_constant / rho)])

        Y_pandas = numpy_to_pandas(self.Y)
        Y_soft_thresholded = pandas_to_numpy(Y_pandas.applymap(lambda x: soft_threshold_for_matrix(x)))
        return elementwise_multiplication(get_matrix_sign(self.W), Y_soft_thresholded)

class VSubProblem():

    def __init__(self, Z_matrix, W_matrix, Lambda_matrix, X_matrix, rho):
        self.rho = rho
        self.Z = Z_matrix
        self.W = W_matrix
        self.Lambda = Lambda_matrix
        self.X = X_matrix

    def compute_new_V_k(self):
        X_transposed = get_matrix_transpose(self.X)







