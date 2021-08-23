import unittest

from ManifoldOptimization.Utils.matrix_operations import *


class MyTestCase(unittest.TestCase):

    def setUp(self):
        print(self._testMethodDoc)

    def test_safe_division_zero_handling(self):
        nominator = np.array([1])
        denominator = np.array([0])
        safe_divison_result = safe_division(nominator, denominator)
        print("==> Showing safe_division_result:")
        print(safe_divison_result)
        self.assertNotEqual(safe_divison_result, np.Inf, msg="Division by zero resulted in infinity")

    def test_safe_division_dividing_non_zero_numbers(self):
        nominator = 1
        denominator = 2
        safe_divison_result = safe_division(nominator, denominator)
        print("==> Showing safe_division_result:")
        print(safe_divison_result)
        self.assertEqual(safe_divison_result, 0.5,
                         msg=f"Division of {nominator} by {denominator} didn't result in {nominator}/{denominator}")

    @staticmethod
    def test_multiply_matrices():
        left_matrix = np.arange(4).reshape(2, 2)
        right_matrix = np.arange(2, 6).reshape(2, 2)
        multiplied_matrices = multiply_matrices(left_matrix, right_matrix)
        print("==> Showing multiplied_matrices:")
        print(multiplied_matrices)
        expected_result = np.array([[4, 5], [16, 21]])
        np.testing.assert_array_equal(multiplied_matrices, expected_result)

    def test_multiply_matrices_input_error(self):
        left_matrix = np.arange(4).reshape(2, 2)
        right_matrix = np.arange(2, 6).reshape(1, 4)
        print("left_matrix:")
        print(left_matrix)
        print("right_matrix:")
        print(right_matrix)
        self.assertRaises(AttributeError, multiply_matrices, left_matrix, right_matrix)

    @staticmethod
    def test_inverse_matrix_non_singular():
        matrix_to_inverse = np.arange(4).reshape(2, 2)
        inversed_matrix = inverse_matrix(matrix_to_inverse)
        print("==> Showing inversed_matrix:")
        print(inversed_matrix)
        expected_result = np.array([[-1.5, 0.5], [1., 0.]])
        np.testing.assert_array_equal(inversed_matrix, expected_result)

    @staticmethod
    def test_inverse_matrix_singular():
        matrix_to_inverse = np.array([[3, 7, 11], [6, 14, 2], [0, 0, 14]])
        inversed_matrix = inverse_matrix(matrix_to_inverse)
        print("==> Showing inversed_matrix:")
        print(inversed_matrix)
        expected_result = np.array([[0.00584708, 0.02293853, -0.00787106],
                                    [0.01364318, 0.05352324, -0.01836582],
                                    [0.02898551, -0.01449275, 0.05072464]])
        np.testing.assert_array_almost_equal(inversed_matrix, expected_result)

    @staticmethod
    def test_transpose_matrix():
        matrix_to_transpose = np.arange(4).reshape(2, 2)
        transposed_matrix = transpose_matrix(matrix_to_transpose)
        print("==> Showing transposed_matrix:")
        print(transposed_matrix)
        expected_result = np.array([[0, 2], [1, 3]])
        np.testing.assert_array_equal(transposed_matrix, expected_result)

    @staticmethod
    def test_matrix_diagonal():
        matrix_to_get_diagonal = np.arange(4).reshape(2, 2)
        matrix_diag = matrix_diagonal(matrix_to_get_diagonal)
        print("==> Showing matrix_diag:")
        print(matrix_diag)
        expected_result = np.array([0, 3])
        np.testing.assert_array_equal(matrix_diag, expected_result)

    @staticmethod
    def test_matrix_sign():
        matrix_to_get_sign = np.array([[13, 0], [132, -14]])
        sign_matrix = matrix_sign(matrix_to_get_sign)
        print("==> Showing sign_matrix:")
        print(sign_matrix)
        expected_result = np.array([[1, 0], [1, -1]])
        np.testing.assert_array_equal(sign_matrix, expected_result)

    @staticmethod
    def test_elementwise_multiplication():
        left_matrix = np.arange(4).reshape(2, 2)
        right_matrix = np.arange(2, 6).reshape(2, 2)
        elementwise_multiplied_matrices = elementwise_multiplication(left_matrix, right_matrix)
        print("==> Showing elementwise_multiplied_matrices:")
        print(elementwise_multiplied_matrices)
        expected_result = np.array([[0, 3], [8, 15]])
        np.testing.assert_array_equal(elementwise_multiplied_matrices, expected_result)

    def test_elementwise_multiplication_input_error(self):
        left_matrix = np.arange(4).reshape(2, 2)
        right_matrix = np.arange(2, 6).reshape(1, 4)
        print("left_matrix:")
        print(left_matrix)
        print("right_matrix:")
        print(right_matrix)
        self.assertRaises(AttributeError, elementwise_multiplication, left_matrix, right_matrix)

    @staticmethod
    def test_vector_into_diagonal_matrix():
        vector = np.array([1, 2, 3])
        diagonal_matrix = vector_into_diagonal_matrix(vector)
        print("==> Showing diagonal_matrix:")
        print(diagonal_matrix)
        expected_result = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        np.testing.assert_array_equal(diagonal_matrix, expected_result)

    def test_frobenius_norm_input_error(self):
        left_matrix = np.arange(4).reshape(2, 2)
        right_matrix = np.arange(2, 6).reshape(1, 4)
        print("left_matrix:")
        print(left_matrix)
        print("right_matrix:")
        print(right_matrix)
        self.assertRaises(AttributeError, frobenius_norm, left_matrix, right_matrix)

    def test_frobenius_norm(self):
        left_matrix = np.arange(4).reshape(2, 2)
        right_matrix = np.arange(2, 6).reshape(2, 2)
        frobenius_norm_of_difference = frobenius_norm(left_matrix, right_matrix)
        print("==> Showing frobenius_norm_of_difference:")
        print(frobenius_norm_of_difference)
        expected_result = 4.0
        self.assertEqual(frobenius_norm_of_difference, expected_result)

    @staticmethod
    def test_soft_thresholding():
        matrix1 = np.arange(4).reshape(2, 2)
        lambda_constant1 = 1
        soft_thresholded_matrix1 = soft_threshold(matrix1, lambda_constant1)
        print("==> Showing soft_thresholded_matrix1:")
        print(soft_thresholded_matrix1)
        expected_result1 = np.array([[0, 0], [1, 2]])

        matrix2 = np.array([[10, 15], [3, 4.5]])
        lambda_constant2 = 4
        soft_thresholded_matrix2 = soft_threshold(matrix2, lambda_constant2)
        print("==> Showing soft_thresholded_matrix2:")
        print(soft_thresholded_matrix2)
        expected_result2 = np.array([[6, 11], [0, 0.5]])
        np.testing.assert_array_equal(soft_thresholded_matrix1, expected_result1)
        np.testing.assert_array_equal(soft_thresholded_matrix2, expected_result2)

    if __name__ == '__main__':
        unittest.main()
