from ManifoldOptimization.Utils.utils import matrix_multiplication

class LambdaSubproblem(lambda_2, V_matrix, X_matrix):
    '''
    This part is just a regular Lasso optimization problem but in a context of matrices from Stiefel manifold.

    Given input matrices we want to compute:
    \Lambda^{k+1} = \argmin _{\boldsymbol{V} \in  \mathcal{V}_{\boldsymbol{p} \times \boldsymbol{p}}^{\boldsymbol{S} \boldsymbol{u}}} \left\|V^{kT}XV^{k}- \Lambda \right\|_{F}^{2} +  \lambda_{2}\|\Lambda\|_{2}
    '''