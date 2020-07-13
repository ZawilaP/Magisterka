import unittest

import numpy as np

from ManifoldOptimization.SFPCA.LambdaSubproblem.lambda_subproblem import LambdaSubproblem


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    @staticmethod
    def test_lambda_subproblem():
        sample_lambda_matrix = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        sample_lambda_2 = 2
        initialized_lambda = LambdaSubproblem(sample_lambda_matrix, sample_lambda_2)
        final_lambda = initialized_lambda.fit()
        print("==> Showing final_lambda:")
        print(final_lambda)
        expected_result = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 2]])
        np.testing.assert_array_equal(final_lambda, expected_result)


if __name__ == '__main__':
    unittest.main()
