import unittest

import numpy as np

from ManifoldOptimization.SFPCA.VSubproblem.madmm_subproblems import WSubproblem, VSubproblem
from ManifoldOptimization.Utils.matrix_operations import vector_into_diagonal_matrix


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_w_subproblem():
        sample_w_matrix = np.array([[1, -1, 3], [11, 13, -11], [-3, 2, 9]])
        sample_lambda = 4
        sample_rho = 2
        initialized_w = WSubproblem(sample_w_matrix, sample_lambda, sample_rho, verbosity=1)
        final_w = initialized_w.fit()
        print("==> Showing final_w")
        print(final_w)
        expected_result = np.array([[0, 0, 1], [9, 11, -9], [-1, 0, 7]])
        np.testing.assert_array_equal(final_w, expected_result)

    @staticmethod
    def test_v_subproblem():
        x_matrix = np.array([[1, -1, 3], [-1, 2, 2], [3, 2, 5]])
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        w_matrix = v_matrix
        z_matrix = np.zeros((3, 3))
        rho = 1 / 2
        initialized_v = VSubproblem(w_matrix, z_matrix, x_matrix, v_matrix, lambda_matrix, rho, verbosity=1)
        final_v = initialized_v.fit()
        print("==> Showing final_v")
        print(final_v)
        expected_result = np.array([[38.591981, 26.723627, 85.750215],
                                    [-7.750934, 11.882393, -0.214772],
                                    [-10.830568, -6.937643, 7.036391]])
        np.testing.assert_array_almost_equal(final_v, expected_result)


if __name__ == '__main__':
    unittest.main()
