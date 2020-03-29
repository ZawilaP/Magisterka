from ManifoldOptimization.Utils.matrix_operations import reconstruct_vector_into_diagonal_matrix, get_matrix_multiplication, get_matrix_transpose
import numpy as np

class SFPCA():

    def __init__(self, madmm_steps, sfpca_steps, input_matrix, rho, lambda_1, lambda_2):
        self.madmm_steps = madmm_steps
        self.sfpca_steps = sfpca_steps
        self.input_matrix = input_matrix
        self.rho = rho
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.X = get_matrix_multiplication(self.input_matrix, get_matrix_transpose(self.input_matrix))
        self.Lambda_matrix, self.V = self.get_spectral_decomposition()

    def get_spectral_decomposition(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.X)
        Lambda_matrix = reconstruct_vector_into_diagonal_matrix(eigenvalues)
        return Lambda_matrix, eigenvectors


