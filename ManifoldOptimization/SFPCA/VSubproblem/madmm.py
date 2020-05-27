import numpy as np

from ManifoldOptimization.SFPCA.VSubproblem.madmm_subproblems import WSubproblem, VSubProblem, ZSubProblem
from ManifoldOptimization.Utils.matrix_operations import get_matrix_inverse, get_matrix_multiplication, \
    get_matrix_transpose


class MADMM():
    '''
    This algorithm has two steps, that require their own algorithms, and one simple step

    W step - similar to how we solved lambda_subproblem

    V step - which requires a lot of of matrix operations

    Z step - simple matrix subtraction and addition

    Initializing all different forms of X computations in this class, so we don't have to compute them at every step
    in it's subproblems.
    '''

    def __init__(self, V_matrix, Lambda_Matrix, X_matrix, rho, lambda_1, n_steps):
        self.V_hat = V_matrix
        self.Lambda = Lambda_Matrix
        self.X = X_matrix
        self.rho = rho
        self.lambda_1 = lambda_1
        self.n_steps = n_steps

        # Initialize variables
        self.V_k = self.V_hat
        self.W_k = self.V_hat
        self.Z_k = np.zeros(self.Lambda.shape(0))

        # Initialize X variables
        self.X_transposed = get_matrix_transpose(self.X)
        self.X_equation = get_matrix_multiplication(self.X, self.X_transposed) - self.X_transposed + self.X
        self.X_part_with_inverse = get_matrix_inverse(
            get_matrix_multiplication(self.X_equation, get_matrix_transpose(self.X_equation)))

        self.new_V = self.execute_madmm()

    def execute_madmm(self):
        """
        Function that executes whole algorithm, for limited number of steps.
        :return: eigenvalues matrix
        """
        step_W = self.W_k
        step_V = self.V_k
        step_Z = self.Z_k
        for step in range(self.n_steps):
            step_V = VSubProblem(step_Z, step_W, self.Lambda, self.X_equation, self.X_part_with_inverse, self.rho)
            step_W = WSubproblem(self.lambda_1, step_V, step_Z, step_W, self.rho)
            step_Z = ZSubProblem(step_Z, step_V, step_W)

        return step_V
