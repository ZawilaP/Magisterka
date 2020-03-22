import numpy as np


def safe_division(nominator, denominator):
    safe_division_constant = 0.0000001
    safe_divided = nominator / (denominator + safe_division_constant)
    return safe_divided


def get_matrix_multiplication(left_matrix, right_matrix):
    return np.dot(left_matrix, right_matrix)


def get_matrix_transpose(matrix_to_transpose: np.array):
    return matrix_to_transpose.transpose()


def get_matrix_diagonal(matrix_to_get_diagonal):
    return np.diagonal(matrix_to_get_diagonal)