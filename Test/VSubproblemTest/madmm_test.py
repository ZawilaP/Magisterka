import unittest

import numpy as np

from ManifoldOptimization.SFPCA.VSubproblem.madmm import WSubproblem, VSubproblem, ZSubproblem, MADMM
from ManifoldOptimization.Utils.matrix_operations import vector_into_diagonal_matrix, multiply_matrices, \
    transpose_matrix


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_w_subproblem():
        x_matrix = np.array([[1, -1, 3], [-1, 1, 2], [3, 2, 1]])
        v_matrix, _, _ = np.linalg.svd(x_matrix)
        z_matrix = np.zeros((3, 3))
        lambda_1 = 0.1
        rho = 1
        initialized_w = WSubproblem(v_matrix, z_matrix, lambda_1, rho, verbosity=3)
        final_w = initialized_w.fit()
        print("==> Showing final_w:")
        print(final_w)
        expected_result = np.array([[-0.510834, -0.492851, -0.424794],
                                    [-0.173369, -0.364142, 0.742521],
                                    [-0.643069, 0.558103, 0.021447]])
        np.testing.assert_array_almost_equal(final_w, expected_result)

    @staticmethod
    def test_v_subproblem():
        x_matrix = np.array([[1, -1, 3], [-1, 1, 2], [3, 2, 1]])
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        w_matrix = v_matrix
        z_matrix = np.zeros((3, 3))
        rho = 1 / 2
        initialized_v = VSubproblem(w_matrix, z_matrix, x_matrix, lambda_matrix, rho, verbosity=3)
        final_v = initialized_v.fit()
        print("==> Showing final_v:")
        print(final_v)
        expected_result = np.array([[-0.610834, 0.45383, -0.648784],
                                    [-0.273369, -0.889917, -0.365126],
                                    [-0.743069, -0.045674, 0.667655]])
        np.testing.assert_array_almost_equal(final_v, expected_result)

    @staticmethod
    def test_z_subproblem():
        v_matrix = np.array([[38.591981, 26.723627, 85.750215],
                             [-7.750934, 11.882393, -0.214772],
                             [-10.830568, -6.937643, 7.036391]])
        w_matrix = np.array([[1, -1, 3], [11, 13, -11], [-3, 2, 9]])
        z_matrix = np.zeros((3, 3))
        initialized_z = ZSubproblem(z_matrix, w_matrix, v_matrix)
        final_z = initialized_z.fit()
        print("==> Showing final_z:")
        print(final_z)
        expected_result = np.array([[37.591981, 27.723627, 82.750215],
                                    [-18.750934, -1.117607, 10.785228],
                                    [-7.830568, -8.937643, -1.963609]])
        np.testing.assert_array_almost_equal(final_z, expected_result)

    @staticmethod
    def test_madmm():
        x_matrix = np.array([[1, -2, 3], [-2, 1, 5], [3, 5, 1]])
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        rho = 10
        lambda_1 = 2
        n_steps = 3
        initialized_v = MADMM(x_matrix, v_matrix, lambda_matrix, lambda_1, rho, n_steps, verbosity=2)
        final_v = initialized_v.fit()
        print("==> Showing final_v:")
        print(final_v)
        final_v_v_t = multiply_matrices(final_v, transpose_matrix(final_v))
        print("==> Final_v: should be orthonormal, showing final_v multiplied by it's transpose:")
        print(final_v_v_t)
        expected_result = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(final_v_v_t, expected_result)


if __name__ == '__main__':
    unittest.main()
