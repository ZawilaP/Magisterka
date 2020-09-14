import unittest

import numpy as np
import pickle as pkl

from ManifoldOptimization.SFPCA.sfpca import SFPCA
from ManifoldOptimization.Utils.matrix_operations import multiply_matrices, transpose_matrix


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_sfpca():
        x_matrix = np.array([[1, 0.82, 0.82, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                             [0.82, 1, 0.82, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                             [0.82, 0.82, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                             [0.001, 0.001, 0.001, 1, 0.92, 0.92, 0.001, 0.001, 0.001],
                             [0.001, 0.001, 0.001, 0.92, 1, 0.92, 0.001, 0.001, 0.001],
                             [0.001, 0.001, 0.001, 0.92, 0.92, 1, 0.001, 0.001, 0.001],
                             [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1, 0.44, 0.44],
                             [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.44, 1, 0.44],
                             [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.44, 0.44, 1]])

        rho = 1
        madmm_steps = 5
        sfpca_steps = 10
        dictionary_of_results = {}

        for lambda_1 in [0.01, 0.1, 0.5, 1, 5]:
            for lambda_2 in [0.01, 0.1, 0.5, 1, 5]:
                sfpca = SFPCA(x_matrix, lambda_1, lambda_2, rho, madmm_steps, sfpca_steps)
                v_final_matrix, lambda_final_matrix = sfpca.fit()
                print(f"==> Lambda_1 = {lambda_1}, Lambda_2 = {lambda_2}")
                print("==> Showing v_final_matrix")
                print(v_final_matrix)
                print("==> Showing lambda_final_matrix")
                print(lambda_final_matrix)
                final_v_v_t = multiply_matrices(v_final_matrix, transpose_matrix(v_final_matrix))
                print("==> Showing final_v_v_t:")
                print(final_v_v_t)
                reconstructed_sparse_x = multiply_matrices(multiply_matrices(v_final_matrix, lambda_final_matrix), transpose_matrix(v_final_matrix))
                print("==> reconstructed_sparse_x:")
                print(reconstructed_sparse_x)
                norm_of_difference = np.linalg.norm(x_matrix - reconstructed_sparse_x, ord = "fro")
                dictionary_of_results[f"lambda_1 = {lambda_1}, lambda_2 = {lambda_2}"] = [v_final_matrix,
                                                                                          lambda_final_matrix,
                                                                                          final_v_v_t,
                                                                                          reconstructed_sparse_x,
                                                                                          norm_of_difference]

        pkl.dump(dictionary_of_results, open("results_of_lambda_grid_search.p", "wb"))
        expected_result = np.array([[38.591981, 26.723627, 85.750215],
                                    [-7.750934, 11.882393, -0.214772],
                                    [-10.830568, -6.937643, 7.036391]])
        np.testing.assert_array_almost_equal(v_final_matrix, expected_result)
        np.testing.assert_array_almost_equal(final_v_v_t, np.identity(final_v_v_t.shape[0]))

if __name__ == '__main__':
    unittest.main()
