import numpy as np
import pandas as pd


def safe_division(nominator: np.array, denominator: np.array) -> np.array:
    """
    Function that safely divides two numbers, so we don't divide by zero
    :param nominator:   'float' nominator of expression
    :param denominator: 'float' denominator of expression
    :return: safely divided nominator by denominator
    """
    safe_division_constant = 0.0000001
    safe_divided = nominator / (denominator + safe_division_constant)
    return safe_divided


def get_matrix_multiplication(left_matrix: np.array, right_matrix: np.array) -> np.array:
    """
    Multiply two matrices
    :param left_matrix:   'np.array' left side of multiplication
    :param right_matrix:  'np.array' right side of multiplication
    :return: multiplied left by right
    """
    return np.dot(left_matrix, right_matrix)


def get_matrix_transpose(matrix_to_transpose: np.array) -> np.array:
    """
    Get transposed matrix
    :param matrix_to_transpose:  'np.array' matrix to be transposed
    :return: transposed matrix
    """
    return matrix_to_transpose.transpose()


def get_matrix_diagonal(matrix_to_get_diagonal: np.array) -> np.array:
    '''
    Function that gets diagonal of a matrix
    :param matrix_to_get_diagonal: 'np.array' from which we extract diagonal
    :return: extracted diagonal of matrix
    '''
    return np.diagonal(matrix_to_get_diagonal)


def get_matrix_sign(matrix_to_get_sign: np.array) -> np.array:
    '''
    Function that gets elementwise signs of matrix elements
    :param matrix_to_get_sign: 'np.array' matrix to get elementwise sign values
    :return: matrix with values from (-1,0,1)
    '''
    return np.sign(matrix_to_get_sign)


def numpy_to_pandas(matrix_to_convert: np.array) -> pd.DataFrame:
    '''
    Convert numpy array to pandas DataFrame
    :param matrix_to_convert: 'np.array' to be converted
    :return: pd.DataFrame from np.array
    '''
    return pd.DataFrame(matrix_to_convert)


def pandas_to_numpy(matrix_to_convert: pd.DataFrame) -> np.array:
    '''
    Convert pandas DataFrame to numpy array
    :param matrix_to_convert:  'pd.DataFrame' to be converted
    :return: np.array from pd.DataFrame
    '''
    return matrix_to_convert.to_numpy()

def elementwise_multiplication(left_matrix: np.array, right_matrix: np.array) -> np.array:
    return np.multiply(left_matrix, right_matrix)


def reconstruct_vector_into_diagonal_matrix(lambda_k: np.array) -> np.array:
    '''
    Transform singular values vector into diagonal matrix
    :param lambda_k: 'np.array' vector to be transformed into matrix
    :return: matrix with diagonal with lambda_k values, 0's otherwise
    '''
    return np.identity(len(lambda_k)) * np.outer(np.ones(len(lambda_k)), lambda_k)
