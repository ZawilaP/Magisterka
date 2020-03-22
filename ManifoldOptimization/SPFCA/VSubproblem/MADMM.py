from ManifoldOptimization.Utils.utils import get_matrix_diagonal, get_matrix_multiplication, get_matrix_transpose, reconstruct_vector_into_diagonal_matrix
import numpy as np

class MADMM():
    '''
    This algorithm has two steps, that require their own algorithms, and one simple step

    W step - similar to how we solved lambda_subproblem

    V step - which requires a lot of of matrix operations

    Z step - simple matrix subtraction and addition
    '''

    def __init__(self):
        print("to be implemented still")