import numpy as np

from ManifoldOptimization.Utils.matrix_operations import numpy_to_pandas, pandas_to_numpy, elementwise_multiplication, \
    get_matrix_sign, get_matrix_transpose, get_matrix_inverse, get_matrix_multiplication


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

#todo: Ta operacja znaku elementwise jest dodgy dla mnie, dopytaj o to

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

    def __init__(self, Z_matrix, W_matrix, Lambda_matrix, X_equation, X_part_with_inverse, rho):
        self.rho = rho
        self.Z = Z_matrix
        self.W = W_matrix
        self.Lambda = Lambda_matrix
        self.X = X_equation
        self.X_with_inverse = X_part_with_inverse
        self.V_k = self.compute_new_V_k()

    def __call__(self, *args, **kwargs):
        return self.V_k

    def compute_new_V_k(self):
        """
        Compute new eigenvalues for the MADMM step
        :return: eigenvalues matrix
        """
        first_brackets = (self.rho / 2) * get_matrix_inverse(np.identity(self.Lambda.shape(0)) + self.Lambda)
        second_brackets = (get_matrix_transpose(W) - get_matrix_transpose(Z))
        return get_matrix_multiplication(
            get_matrix_multiplication(get_matrix_multiplication(first_brackets, second_brackets),
                                      get_matrix_transpose(self.X)), self.X_with_inverse)


class ZSubProblem():

    def __init__(self, Z_matrix, V_matrix, W_matrix):
        self.Z_k = Z_matrix + V_matrix - W_matrix

    def __call__(self, *args, **kwargs):
        return self.Z_k
