from ManifoldOptimization.Utils.utils import get_matrix_diagonal, get_matrix_multiplication, get_matrix_transpose, reconstruct_vector_into_diagonal_matrix
import numpy as np

class LambdaSubproblem():
    '''
    This part is just a regular Lasso optimization problem but in a context of matrices from Stiefel manifold.

    Given input matrices we want to compute:
    \Lambda^{k+1} = \argmin _{\boldsymbol{V} \in  \mathcal{V}_{\boldsymbol{p} \times \boldsymbol{p}}^{\boldsymbol{S} \boldsymbol{u}}} \left\|V^{kT}XV^{k}- \Lambda \right\|_{F}^{2} +  \lambda_{2}\|\Lambda\|_{2}

    This problem has nice analytical solution, which we use here to get Lambda^{K+1}
    '''

    def __init__(self, lambda_2, V_matrix, X_matrix):
        self.lambda_constant = lambda_2
        self.V = V_matrix
        self.X = X_matrix
        self.Z = self.compute_z()
        self.lambda_k = self.compute_new_lambda_k()
        self.reconstructed_lambda_k = reconstruct_vector_into_diagonal_matrix(self.lambda_k)

    def __call__(self, *args, **kwargs):
        return self.reconstructed_lambda_k

    def compute_z(self):
        '''
        Compute value of Z = V^{KT}XV^{K}
        :return: computed value of Z
        '''
        V_transposed = get_matrix_transpose(self.V)
        Z = get_matrix_multiplication(get_matrix_multiplication(V_transposed, X), self.V)
        return Z

    def compute_new_lambda_k(self):
        '''
        Apply soft thresholding operator to Lasso problem
        :return: vector of new singular values
        '''
        lambda_constant = self.lambda_constant

        def soft_threshold_operator(Z_element):
            return max([0, Z_element - (lambda_constant / 2)])

        Z_diagonal = get_matrix_diagonal(self.Z)
        lambda_k = map(soft_threshold_operator, Z_diagonal)
        return np.array(lambda_k)


