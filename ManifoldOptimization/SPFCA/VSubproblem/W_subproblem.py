from ManifoldOptimization.Utils.utils import get_matrix_diagonal, get_matrix_multiplication, get_matrix_transpose, reconstruct_vector_into_diagonal_matrix
import numpy as np

class WSubproblem():

    def __init__(self, lambda_1, V_matrix, Z_matrix, W_matrix, rho):
        self.lambda_constant = lambda_1
        self.V = V_matrix
        self.Z = Z_matrix
        self.W = W_matrix
        self.rho = rho
        self.Y = self.V + self.Z

    def soft_threshold_for_matrix(self, element):
        return max([0, element - (self.lambda_constant/self.rho)])



    def compute_new_W_k(self):

