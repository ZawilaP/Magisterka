from ManifoldOptimization.Utils.matrix_operations import soft_threshold


class LambdaSubproblem():
    """
    Compute Lasso optimization problem for Lambda matrix, which is defined as:
    $$\hat{\Lambda} = \argmin _{\Lambda \in \mathbb{R}_{+}^{p}} \left\|X-V \Lambda V^{\top}\right\|_{F}^{2} + \lambda_{2}\|\Lambda\|_{2}$$
    Which simplifies to soft-thresholding Lambda matrix by lambda_2 regularization parameter, divided by two
    """

    def __init__(self, lambda_matrix, lambda_2):
        self.lambda_matrix = lambda_matrix
        self.lambda_2 = lambda_2

    def fit(self):
        division_constant = 2
        new_lambda = soft_threshold(self.lambda_matrix, self.lambda_2 / division_constant)
        return new_lambda
