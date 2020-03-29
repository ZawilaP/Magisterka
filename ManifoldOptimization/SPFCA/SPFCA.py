import numpy as np

class SFPCA():

    def __init__(self, madmm_steps, sfpca_steps, X_matrix, rho, lambda_1, lambda_2):
        self.madmm_steps = madmm_steps
        self.sfpca_steps = sfpca_steps
        self.X = X_matrix
        self.rho = rho
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2



