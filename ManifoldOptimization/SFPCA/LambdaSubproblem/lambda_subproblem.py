import numpy as np

from ManifoldOptimization.Utils.matrix_operations import soft_threshold, multiply_matrices, transpose_matrix, matrix_diagonal, vector_into_diagonal_matrix


class LambdaSubproblem():
    """
    Compute Lasso optimization problem for Lambda matrix, which is defined as:
    $$\hat{\Lambda} = \argmin _{\Lambda \in \mathbb{R}_{+}^{p}} \left\|X-V \Lambda V^{\top}\right\|_{F}^{2} + \lambda_{2}\|\Lambda\|_{2}$$
    Which simplifies to soft-thresholding V^T*X*V by lambda_2 regularization parameter, divided by two
    """

    def __init__(self, x_matrix: np.array, v_matrix: np.array, lambda_2: float, verbosity: int = 0):
        self.x_matrix = x_matrix
        self.v_matrix = v_matrix
        self.lambda_2 = lambda_2
        self.verbosity = verbosity

    # todo: Zoptymalizuj np.diag obliczenia
    def fit(self):
        division_constant = 2
        v_transposed_x_v = multiply_matrices(multiply_matrices(transpose_matrix(self.v_matrix), self.x_matrix),
                                             self.v_matrix)
        if self.verbosity > 2:
            print("==> LambdaSubproblem ==> Showing v_transposed_x_v")
            print(v_transposed_x_v)
        new_lambda = vector_into_diagonal_matrix(soft_threshold(matrix_diagonal(v_transposed_x_v), self.lambda_2 / division_constant))
        return new_lambda
