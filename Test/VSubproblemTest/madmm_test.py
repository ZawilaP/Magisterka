import unittest

import numpy as np

from ManifoldOptimization.SFPCA.VSubproblem.madmm import WSubproblem, VSubproblem, ZSubproblem, MADMM
from ManifoldOptimization.Utils.matrix_operations import vector_into_diagonal_matrix


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_w_subproblem():
        w_matrix = np.array([[1, -1, 3], [11, 13, -11], [-3, 2, 9]])
        lambda_1 = 4
        rho = 2
        initialized_w = WSubproblem(w_matrix, lambda_1, rho, verbosity=3)
        final_w = initialized_w.fit()
        print("==> Showing final_w")
        print(final_w)
        expected_result = np.array([[0, 0, 1], [9, 11, -9], [-1, 0, 7]])
        np.testing.assert_array_equal(final_w, expected_result)

    @staticmethod
    def test_v_subproblem():
        x_matrix = np.array([[1, -1, 3], [-1, 1, 2], [3, 2, 1]])
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        w_matrix = v_matrix
        z_matrix = np.zeros((3, 3))
        rho = 1 / 2
        initialized_v = VSubproblem(w_matrix, z_matrix, x_matrix, v_matrix, lambda_matrix, rho, verbosity=3)
        final_v = initialized_v.fit()
        print("==> Showing final_v")
        print(final_v)
        expected_result = np.array([[ 21.875269,   9.789922,  26.610868],
       [-34.176691, -26.756848,  37.938325],
       [  4.096128,  -6.576053,  -0.947916]])
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
        print("==> Showing final_z")
        print(final_z)
        expected_result = np.array([[37.591981, 27.723627, 82.750215],
                                    [-18.750934, -1.117607, 10.785228],
                                    [-7.830568, -8.937643, -1.963609]])
        np.testing.assert_array_almost_equal(final_z, expected_result)

    @staticmethod
    def test_madmm():
        x_matrix = np.array([[1, -1, 3], [-1, 2, 2], [3, 2, 5]])
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        rho = 10
        lambda_1 = 2
        n_steps = 3
        initialized_v = MADMM(x_matrix, v_matrix, lambda_matrix, lambda_1, rho, n_steps, verbosity=2)
        final_v = initialized_v.fit()
        print("==> Showing final_v")
        print(final_v)
        expected_result = np.array([[38.591981, 26.723627, 85.750215],
                                    [-7.750934, 11.882393, -0.214772],
                                    [-10.830568, -6.937643, 7.036391]])
        np.testing.assert_array_almost_equal(final_v, expected_result)


if __name__ == '__main__':
    unittest.main()
