import numpy as np

from ManifoldOptimization.SFPCA.LambdaSubproblem.lambda_subproblem import LambdaSubproblem
from ManifoldOptimization.SFPCA.VSubproblem.madmm import MADMM
from ManifoldOptimization.Utils.matrix_operations import vector_into_diagonal_matrix, multiply_matrices, transpose_matrix


class SFPCA():
    """
    Calculate SFPCA
    """
    def __init__(self, x_matrix, lambda_1, lambda_2, rho, madmm_steps, sfpca_steps, verbosity):
        self.x_matrix = x_matrix
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = rho
        self.madmm_steps = madmm_steps
        self.sfpca_steps = sfpca_steps
        self.verbosity = verbosity

    def fit(self):
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(self.x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        for i in range(self.sfpca_steps):
            if self.verbosity > 0:
                print("\n")
                print(f"=============== SFPCA step {i} ================")
            v_matrix = MADMM(self.x_matrix, v_matrix, lambda_matrix, self.lambda_1, self.rho, self.madmm_steps, self.verbosity).fit()
            if self.verbosity > 0:
                print(f"==> SFPCA ==> Showing v_matrix from step {i}")
                print(v_matrix)
            lambda_matrix = LambdaSubproblem(self.x_matrix, v_matrix, self.lambda_2).fit()
            if self.verbosity > 0:
                print(f"==> SFPCA ==> Showing lambda_matrix from step {i}")
                print(lambda_matrix)
        if self.verbosity > 0:
            print(f"==> SFPCA ==> Showing final v_matrix")
            print(v_matrix)
            print(f"==> SFPCA ==> Showing final lambda_matrix")
            print(lambda_matrix)
        return multiply_matrices(multiply_matrices(v_matrix, lambda_matrix), transpose_matrix(v_matrix))
