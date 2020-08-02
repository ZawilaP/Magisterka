import numpy as np

from ManifoldOptimization.SFPCA.LambdaSubproblem.lambda_subproblem import LambdaSubproblem
from ManifoldOptimization.SFPCA.VSubproblem.madmm import MADMM
from ManifoldOptimization.Utils.matrix_operations import vector_into_diagonal_matrix, multiply_matrices, transpose_matrix


class SFPCA():
    """
    Calculate SFPCA
    """
    def __init__(self, x_matrix, lambda_1, lambda_2, rho, madmm_steps, sfpca_steps, verbosity: int = -1):
        self.x_matrix = x_matrix
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = rho
        self.madmm_steps = madmm_steps
        self.sfpca_steps = sfpca_steps
        self.verbosity = verbosity

    # todo: Wyznaczyc minimalna lambda_1 co wyzeruje nam macierz V (dla jakich lambd 0 spelnia warunek optymalnosci na Stiefelu)
    # todo: W ogolnym problemnie na V, a nie podkrokach w MADMM
    # todo: Zeby ograniczyc do pierwszych k wartosci szczegolnych i wektorow szczegolnych (KOLUMNOWO)

    # todo: Dodaj progowanie wartości i wtedy moze cross-validacja na Grassmanianie byc dobrym wyznacznikiem znajdowania V
    # todo: Kryterium stopu z jakimś poziomem tolerancji
    def fit(self):
        v_matrix, lambda_matrix_vector, _ = np.linalg.svd(self.x_matrix)
        lambda_matrix = vector_into_diagonal_matrix(lambda_matrix_vector)
        for i in range(self.sfpca_steps):
            if self.verbosity > -1:
                print("\n")
                print(f"=============== SFPCA step {i} ================")
            v_matrix = MADMM(x_matrix=self.x_matrix,
                             v_matrix=v_matrix,
                             lambda_matrix=lambda_matrix,
                             lambda_1=self.lambda_1,
                             sfpca_steps=self.sfpca_steps,
                             current_sfpca_step=i,
                             rho=self.rho,
                             n_steps=self.madmm_steps,
                             verbosity=self.verbosity).fit()
            if self.verbosity > -1:
                print(f"==> SFPCA ==> Showing v_matrix from step {i}")
                print(v_matrix)

            if i < self.sfpca_steps - 1:
                lambda_matrix = LambdaSubproblem(x_matrix=self.x_matrix,
                                                 v_matrix=v_matrix,
                                                 lambda_2=self.lambda_2).fit()
            if self.verbosity > -1:
                print(f"==> SFPCA ==> Showing lambda_matrix from step {i}")
                print(lambda_matrix)
        if self.verbosity > -1:
            print(f"==> SFPCA ==> Showing final v_matrix")
            print(v_matrix)
            print(f"==> SFPCA ==> Showing final lambda_matrix")
            print(lambda_matrix)
        return v_matrix, lambda_matrix
