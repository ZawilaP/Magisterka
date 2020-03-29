from ManifoldOptimization.Utils.matrix_operations import get_matrix_diagonal, get_matrix_multiplication, get_matrix_transpose, reconstruct_vector_into_diagonal_matrix
import numpy as np

class MADMM():
    '''
    This algorithm has two steps, that require their own algorithms, and one simple step

    W step - similar to how we solved lambda_subproblem

    V step - which requires a lot of of matrix operations

    Z step - simple matrix subtraction and addition
    '''

    def __init__(self, Lambda_Matrix, X_matrix, rho, lambda_1):
        self.V_hat = V_matrix
        self.Lambda = Lambda_Matrix
        self.X = X_matrix
        self.rho = rho
        self.lambda_1 = lambda_1
        self.initialize_variables()

    def initialize_variables(self):
        self.V_k = self.V_hat
        self.W_k = self.V_hat
        self.Z_k = np.zeros(self.Lambda.shape(0))
