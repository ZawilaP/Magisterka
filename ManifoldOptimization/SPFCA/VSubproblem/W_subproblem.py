from ManifoldOptimization.Utils.utils import numpy_to_pandas, pandas_to_numpy, elementwise_multiplication, get_matrix_sign
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
        rho = self.rho
        lambda_constant = self.lambda_constant

        def soft_threshold_for_matrix(element):
            return max([0, element - (lambda_constant / rho)])

        Y_pandas = numpy_to_pandas(self.Y)
        Y_soft_thresholded = pandas_to_numpy(Y_pandas.applymap(lambda x: soft_threshold_for_matrix(x)))
        return elementwise_multiplication(get_matrix_sign(self.W), Y_soft_thresholded)





