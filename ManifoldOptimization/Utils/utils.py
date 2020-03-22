import numpy as np


def safe_division(nominator, denominator):
    """
    Function that safely divides two numbers, so we don't divide by zero
    :param nominator: nominator of expression
    :param denominator: denominator of expression
    :return: safely divided nominator by denominator
    """
    safe_division_constant = 0.0000001
    safe_divided = nominator / (denominator + safe_division_constant)
    return safe_divided


def get_matrix_multiplication(left_matrix, right_matrix):
    """
    Multiply two matrices
    :param left_matrix: left side of multiplication
    :param right_matrix: right side of multiplication
    :return: multiplied left by right
    """
    return np.dot(left_matrix, right_matrix)


def get_matrix_transpose(matrix_to_transpose: np.array):
    return matrix_to_transpose.transpose()


def get_matrix_diagonal(matrix_to_get_diagonal):
    return np.diagonal(matrix_to_get_diagonal)