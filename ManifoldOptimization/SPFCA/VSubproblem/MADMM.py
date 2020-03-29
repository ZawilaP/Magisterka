from ManifoldOptimization.Utils.matrix_operations import get_matrix_diagonal, get_matrix_multiplication, get_matrix_transpose, reconstruct_vector_into_diagonal_matrix
from ManifoldOptimization.SPFCA.VSubproblem.MADMM_Subproblems import WSubproblem, VSubProblem, ZSubProblem
import numpy as np

class MADMM():
    '''
    This algorithm has two steps, that require their own algorithms, and one simple step

    W step - similar to how we solved lambda_subproblem

    V step - which requires a lot of of matrix operations

    Z step - simple matrix subtraction and addition
    '''

    def __init__(self, Lambda_Matrix, X_matrix, rho, lambda_1, n_steps):
        self.V_hat = V_matrix
        self.Lambda = Lambda_Matrix
        self.X = X_matrix
        self.rho = rho
        self.lambda_1 = lambda_1
        self.n_steps = n_steps
        self.initialize_variables()
        self.new_V = self.execute_MADMM()

    def initialize_variables(self):
        self.V_k = self.V_hat
        self.W_k = self.V_hat
        self.Z_k = np.zeros(self.Lambda.shape(0))

    def execute_MADMM(self):
        step_W = self.W_k
        step_V = self.V_k
        step_Z = self.Z_k
        for step in self.n_steps:
            step_V = VSubProblem(step_Z, step_W, self.Lambda, self.X_matrix, self.rho)
            step_W = WSubproblem(self.lambda_1, step_V, step_Z, step_W, self.rho)
            step_Z = ZSubProblem(step_Z, step_V, step_W)

        return step_V