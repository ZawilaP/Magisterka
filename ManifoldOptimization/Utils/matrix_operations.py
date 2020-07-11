import numpy as np
import pandas as pd


def safe_division(nominator: np.array, denominator: np.array) -> np.array:
    """
    Function that safely divides two arrays, so we don't divide by zero
    :param nominator:   'float' nominator of expression
    :param denominator: 'float' denominator of expression
    :return: safely divided nominator by denominator
    """
    safe_division_constant = 0.000001
    safe_divided = (nominator / (denominator + safe_division_constant)) if (
        np.any(denominator == 0)) else (nominator / denominator)
    return safe_divided


def multiply_matrices(left_matrix: np.array, right_matrix: np.array) -> np.array:
    """
    Multiply two matrices
    :param left_matrix:   'np.array' left side of multiplication
    :param right_matrix:  'np.array' right side of multiplication
    :return: multiplied left by right
    """
    if left_matrix.shape != right_matrix.shape:
        raise AttributeError("Both matrices need to be of the same shape")
    return np.dot(left_matrix, right_matrix)


def inverse_matrix(matrix_to_inverse: np.array) -> np.array:
    '''
    Get inverse of matrix, in case of singular matrix compute pseudo-inverse
    :param matrix_to_inverse: 'np.array' matrix to get inverse
    :return: inversed matrix
    '''
    try:
        return np.linalg.inv(matrix_to_inverse)
    except np.linalg.LinAlgError:
        print("Matrix non-invertible, using moore-penrose pseudo-inverse")
        return np.linalg.pinv(matrix_to_inverse)


def transpose_matrix(matrix_to_transpose: np.array) -> np.array:
    """
    Get transposed matrix
    :param matrix_to_transpose:  'np.array' matrix to be transposed
    :return: transposed matrix
    """
    return matrix_to_transpose.transpose()


def matrix_diagonal(matrix_to_get_diagonal: np.array) -> np.array:
    '''
    Function that gets diagonal of a matrix
    :param matrix_to_get_diagonal: 'np.array' from which we extract diagonal
    :return: extracted diagonal of matrix
    '''
    return np.diagonal(matrix_to_get_diagonal)


def matrix_sign(matrix_to_get_sign: np.array) -> np.array:
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
    '''
    Perform elementwise multiplication of two matrices
    :param left_matrix:  'np.array'
    :param right_matrix: 'np.array'
    :return: matrix from multiplication
    '''
    if left_matrix.shape != right_matrix.shape:
        raise AttributeError("Both matrices need to be of the same shape")
    return np.multiply(left_matrix, right_matrix)


def vector_into_diagonal_matrix(vector: np.array) -> np.array:
    '''
    Transform singular values vector into diagonal matrix
    :param vector: 'np.array' vector to be transformed into matrix
    :return: matrix with diagonal with lambda_k values, 0's otherwise
    '''
    return np.identity(len(vector)) * np.outer(np.ones(len(vector)), vector)


def frobenius_norm(left_matrix: np.array, right_matrix: np.array) -> float:
    """
    Calculate frobenius norm of difference of two matrices
    :param left_matrix: 'np.array'
    :param right_matrix: 'np.array'
    :return:
    """
    if left_matrix.shape != right_matrix.shape:
        raise AttributeError("Both matrices need to be of the same shape")
    matrix_difference = left_matrix - right_matrix
    return np.linalg.norm(matrix_difference, ord="fro")
