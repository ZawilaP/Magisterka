from ManifoldOptimization.Utils.utils import get_matrix_diagonal, get_matrix_multiplication, get_matrix_transpose, reconstruct_vector_into_diagonal_matrix
import numpy as np

class WSubproblem():

    def __init__(self, lambda_1, V_matrix, Z_matrix, rho):
        self.lambda_constant = lambda_1
        self.V = V_matrix
        self.Z = Z_matrix
        self.rho = rho

