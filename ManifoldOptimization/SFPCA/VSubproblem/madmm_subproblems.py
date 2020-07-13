import numpy as np

from ManifoldOptimization.Utils.matrix_operations import matrix_sign, soft_threshold, elementwise_multiplication, \
    transpose_matrix, multiply_matrices


class WSubproblem():
    """
    Calculate solution of the W subproblem of MADMM:
    $$
    W^{k+1} = \argmin _{W \in \mathbb{R}^{p \times p}}  \ \lambda_{1}\|W\|_{1} + \frac{\rho}{2} \left\|V^{k+1} - W + Z^{k} \right\|_{F}^{2}
    $$
    """

    def __init__(self, w_matrix, lambda_1, rho, verbosity=0):
        self.rho = rho
        self.w_matrix = w_matrix
        self.lambda_1 = lambda_1
        self.verbosity = verbosity

    def fit(self):
        w_matrix_sign = matrix_sign(self.w_matrix)
        if self.verbosity > 0:
            print("w_matrix_sign:")
            print(w_matrix_sign)
        w_matrix_absolute = np.abs(self.w_matrix)
        if self.verbosity > 0:
            print("w_matrix_absolute:")
            print(w_matrix_absolute)
        w_matrix_thresholded = soft_threshold(w_matrix_absolute, self.lambda_1 / self.rho)
        if self.verbosity > 0:
            print("w_matrix_thresholded:")
            print(w_matrix_thresholded)
        return elementwise_multiplication(w_matrix_sign, w_matrix_thresholded)


class VSubproblem():
    """
    Calculate solution of the V subproblem of MADMM:
    $$
    V^{k+1} = \argmin _{\boldsymbol{V} \in  \mathcal{V}_{\boldsymbol{p} \times \boldsymbol{p}}^{\boldsymbol{S} \boldsymbol{u}}} \left\|X-V \Lambda V^{\top}\right\|_{F}^{2} + \frac{\rho}{2} \left\|V^{k+1} - W + Z^{k} \right\|_{F}^{2}
    $$
    """

    def __init__(self, w_matrix, z_matrix, x_matrix, v_matrix, lambda_matrix, rho, verbosity=0):
        self.w_matrix = w_matrix
        self.z_matrix = z_matrix
        self.x_matrix = x_matrix
        self.v_matrix = v_matrix
        self.lambda_matrix = lambda_matrix
        self.rho = rho
        self.verbosity = verbosity

    def fit(self):
        x_transposed = transpose_matrix(self.x_matrix)
        if self.verbosity > 0:
            print("x_transposed:")
            print(x_transposed)
        v_transposed = transpose_matrix(self.v_matrix)
        if self.verbosity > 0:
            print("v_transposed:")
            print(v_transposed)
        w_transposed = transpose_matrix(self.w_matrix)
        if self.verbosity > 0:
            print("w_transposed:")
            print(w_transposed)
        z_transposed = transpose_matrix(self.z_matrix)
        if self.verbosity > 0:
            print("z_transposed:")
            print(z_transposed)
        x_x_transposed = multiply_matrices(self.x_matrix, x_transposed)
        if self.verbosity > 0:
            print("x_x_transposed:")
            print(x_x_transposed)
        x_transposed_sum_x = x_transposed + self.x_matrix
        if self.verbosity > 0:
            print("x_transposed_sum_x:")
            print(x_transposed_sum_x)
        double_v_transposed_x_x_transposed = 2 * multiply_matrices(v_transposed, x_x_transposed)
        if self.verbosity > 0:
            print("double_v_transposed_x_x_transposed:")
            print(double_v_transposed_x_x_transposed)
            print("lambda_matrix:")
            print(self.lambda_matrix)
        double_lambda_v_transposed = 2 * multiply_matrices(self.lambda_matrix, v_transposed)
        if self.verbosity > 0:
            print("double_lambda_v_transposed:")
            print(double_lambda_v_transposed)
        double_lambda_v_transposed_x_transposed_sum_x = multiply_matrices(double_lambda_v_transposed,
                                                                          x_transposed_sum_x)
        if self.verbosity > 0:
            print("double_lambda_v_transposed_x_transposed_sum_x:")
            print(double_lambda_v_transposed_x_transposed_sum_x)
        lambdas_v_and_x_part = double_v_transposed_x_x_transposed - double_lambda_v_transposed_x_transposed_sum_x
        if self.verbosity > 0:
            print("lambdas_v_and_x_part:")
            print(lambdas_v_and_x_part)
        return lambdas_v_and_x_part - self.rho * w_transposed + self.rho * z_transposed
