import unittest

import numpy as np

from ManifoldOptimization.SFPCA.LambdaSubproblem.lambda_subproblem import LambdaSubproblem


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_lambda_subproblem():
        x_matrix = np.array([[1, -1, 3], [-1, 1, 2], [3, 2, 1]])
        v_matrix, _, _ = np.linalg.svd(x_matrix)
        sample_lambda_2 = 2
        verbosity = 2
        initialized_lambda = LambdaSubproblem(x_matrix, v_matrix, sample_lambda_2, verbosity)
        final_lambda = initialized_lambda.fit()
        print("==> Showing final_lambda:")
        print(final_lambda)
        expected_result = np.array([[3.201912, 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.911179]])
        np.testing.assert_array_almost_equal(final_lambda, expected_result)


if __name__ == '__main__':
    unittest.main()
