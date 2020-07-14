import unittest

import numpy as np

from ManifoldOptimization.SFPCA.sfpca import SFPCA


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_sfpca():
        x_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        lambda_1 = 2
        lambda_2 = 2
        rho = 1
        madmm_steps = 3
        sfpca_steps = 4
        verbosity = 1
        sfpca = SFPCA(x_matrix, lambda_1, lambda_2, rho, madmm_steps, sfpca_steps, verbosity)
        final_matrix = sfpca.fit()
        print("==> Showing final_matrix")
        print(final_matrix)
        expected_result = np.array([[38.591981, 26.723627, 85.750215],
                                    [-7.750934, 11.882393, -0.214772],
                                    [-10.830568, -6.937643, 7.036391]])
        np.testing.assert_array_almost_equal(final_matrix, expected_result)

if __name__ == '__main__':
    unittest.main()
