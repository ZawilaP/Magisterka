from ManifoldOptimization.Utils.matrix_operations import reconstruct_vector_into_diagonal_matrix, get_matrix_multiplication, get_matrix_transpose
from ManifoldOptimization.SFPCA.LambdaSubproblem.lambda_subproblem import LambdaSubproblem
from ManifoldOptimization.SFPCA.VSubproblem.madmm import MADMM
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

    def __call__(self, *args, **kwargs):
        return self.execute_sfpca()

    def get_spectral_decomposition(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.X)
        Lambda_matrix = reconstruct_vector_into_diagonal_matrix(eigenvalues)
        return Lambda_matrix, eigenvectors

    def execute_sfpca(self):
        step_lambda, step_v = self.get_spectral_decomposition()
        for step in range(self.sfpca_steps):
            step_lambda = LambdaSubproblem(self.lambda_2, step_v, self.X)
            step_v = MADMM(step_lambda, self.X, self.rho, self.lambda_1, self.madmm_steps)

        return step_lambda, step_v


