import numpy as np
import tensorflow as tf
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent

from ManifoldOptimization.Utils.matrix_operations import matrix_sign, soft_threshold, elementwise_multiplication


class MADMM():
    """
    Calculate MADMM for V subproblem of SFPCA
    """

    def __init__(self, x_matrix: np.array, v_matrix: np.array, lambda_matrix: np.array, lambda_1: float, sfpca_steps: int,
                 current_sfpca_step: int, rho: float = 1, n_steps: int = 100, verbosity: int = 0):
        self.x_matrix = x_matrix
        self.v_matrix = v_matrix
        self.lambda_matrix = lambda_matrix
        self.lambda_1 = lambda_1
        self.rho = rho
        self.n_steps = n_steps
        self.sfpca_steps = sfpca_steps
        self.current_sfpca_step = current_sfpca_step
        self.verbosity = verbosity

    def fit(self):
        w_matrix = self.v_matrix
        z_matrix = np.zeros((w_matrix.shape[0], w_matrix.shape[1]))
        for i in range(self.n_steps):
            if self.verbosity > 1:
                print("\n")
                print(f"=============== MADMM step {i} ===============")
            v_matrix = VSubproblem(w_matrix, z_matrix, self.x_matrix, self.lambda_matrix, self.rho,
                                   self.verbosity).fit()
            if self.verbosity > 1:
                print(f"==> MADMM ==> Showing v_matrix from step {i}:")
                print(v_matrix)

            w_matrix = WSubproblem(v_matrix, z_matrix, self.lambda_1, self.rho, self.verbosity).fit()
            if self.verbosity > 1:
                print(f"==> MADMM ==> Showing w_matrix from step {i}:")
                print(w_matrix)

            z_matrix = ZSubproblem(z_matrix, w_matrix, v_matrix).fit()
            if self.verbosity > 1:
                print(f"==> MADMM ==> Showing z_matrix from step {i}:")
                print(z_matrix)

        if self.verbosity > 1:
            print("==> MADMM ==> Showing final v_matrix:")
            print(w_matrix)

        returned_matrix = v_matrix if (self.current_sfpca_step < self.sfpca_steps - 1) else w_matrix

        return returned_matrix


class WSubproblem():
    """
    Calculate solution of the W subproblem of MADMM:
    $$
    W^{k+1} = \argmin _{W \in \mathbb{R}^{p \times p}}  \ \lambda_{1}\|W\|_{1} + \frac{\rho}{2} \left\|V^{k+1} - W + Z^{k} \right\|_{F}^{2}
    $$
    """

    def __init__(self, v_matrix: np.array, z_matrix: np.array, lambda_1: float, rho: float, verbosity: int = 0):
        self.rho = rho
        self.v_matrix = v_matrix
        self.z_matrix = z_matrix
        self.lambda_1 = lambda_1
        self.verbosity = verbosity

    def fit(self):
        v_sum_z = self.v_matrix + self.z_matrix
        v_sum_z_matrix_sign = matrix_sign(v_sum_z)
        if self.verbosity > 2:
            print("==> WSubproblem ==> Showing v_sum_z_matrix_sign:")
            print(v_sum_z_matrix_sign)

        v_sum_z_matrix_absolute = np.abs(v_sum_z)
        if self.verbosity > 2:
            print("==> WSubproblem ==> Showing v_sum_z_matrix_absolute:")
            print(v_sum_z_matrix_absolute)

        v_matrix_thresholded = soft_threshold(v_sum_z_matrix_absolute, self.lambda_1 / self.rho)
        if self.verbosity > 2:
            print("==> WSubproblem ==> Showing v_matrix_thresholded:")
            print(v_matrix_thresholded)
        return elementwise_multiplication(v_sum_z_matrix_sign, v_matrix_thresholded)


class VSubproblem():
    """
    Calculate solution of the V subproblem of MADMM:
    $$
    V^{k+1} = \argmin _{\boldsymbol{V} \in  \mathcal{V}_{\boldsymbol{p} \times \boldsymbol{p}}^{\boldsymbol{S} \boldsymbol{u}}} \left\|X-V \Lambda V^{\top}\right\|_{F}^{2} + \frac{\rho}{2} \left\|V^{k+1} - W + Z^{k} \right\|_{F}^{2}
    $$
    """

    def __init__(self, w_matrix: np.array, z_matrix: np.array, x_matrix: np.array,
                 lambda_matrix: np.array, rho: float, verbosity: int = 0):
        self.w_matrix = w_matrix
        self.z_matrix = z_matrix
        self.x_matrix = x_matrix
        self.lambda_matrix = lambda_matrix
        self.rho = rho
        self.verbosity = verbosity

    def fit(self):
        v_matrix_shape = (self.w_matrix.shape[0], self.w_matrix.shape[1])
        w_matrix = tf.convert_to_tensor(self.w_matrix, dtype=tf.float64)
        z_matrix = tf.convert_to_tensor(self.z_matrix, dtype=tf.float64)
        x_matrix = tf.convert_to_tensor(self.x_matrix, dtype=tf.float64)
        lambda_matrix = tf.convert_to_tensor(self.lambda_matrix, dtype=tf.float64)
        x = tf.Variable(initial_value=tf.ones(v_matrix_shape, dtype=tf.dtypes.float64))

        cost = tf.norm(x_matrix - tf.linalg.matmul(tf.linalg.matmul(x, lambda_matrix), tf.transpose(x))) + self.rho / 2 * tf.norm(x - w_matrix + z_matrix)

        manifold = Stiefel(v_matrix_shape[0], v_matrix_shape[1])
        problem = Problem(manifold=manifold, cost=cost, arg=x)
        solver = SteepestDescent(logverbosity=self.verbosity)
        if self.verbosity > 2:
            v_optimal, _ = solver.solve(problem)
        else:
            v_optimal = solver.solve(problem)

        if self.verbosity > 2:
            print("==> WSubproblem ==> Showing v_optimal:")
            print(v_optimal)

        return v_optimal


class ZSubproblem():
    """
    Calculate solution of Z suproblem of MADMM
    """

    def __init__(self, z_matrix: np.array, w_matrix: np.array, v_matrix: np.array):
        self.z_matrix = z_matrix
        self.w_matrix = w_matrix
        self.v_matrix = v_matrix

    def fit(self):
        return self.z_matrix + self.v_matrix - self.w_matrix
