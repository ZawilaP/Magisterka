import datetime
import pickle as pkl

import numpy as np

from ManifoldOptimization.SFPCA.sfpca import SFPCA
from ManifoldOptimization.Utils.matrix_operations import multiply_matrices, transpose_matrix

x_matrix = np.array([[1, 0.82, 0.82, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.82, 1, 0.82, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.82, 0.82, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 1, 0.92, 0.92, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.92, 1, 0.92, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.92, 0.92, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1, 0.44, 0.44, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.44, 1, 0.44, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.44, 0.44, 1, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1, 0.66, 0.66],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.66, 1, 0.66],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.66, 0.66, 1]])

rho = 0.1
madmm_steps_range = [3, 5, 10, 20]
sfpca_steps_range = [3, 5, 10, 20]
dictionary_of_results = {}
lambda_1 = 0.009
lambda_2 = 1.0

for sfpca_steps in sfpca_steps_range:
    for madmm_steps in madmm_steps_range:
        sfpca = SFPCA(x_matrix, lambda_1, lambda_2, rho, madmm_steps, sfpca_steps, verbosity=10)
        v_final_matrix, lambda_final_matrix = sfpca.fit()
        print(f"==> Lambda_1 = {lambda_1}, Lambda_2 = {lambda_2}")
        print("==> Showing v_final_matrix")
        print(v_final_matrix)
        print("==> Showing lambda_final_matrix")
        print(lambda_final_matrix)
        final_v_v_t = multiply_matrices(v_final_matrix, transpose_matrix(v_final_matrix))
        print("==> Showing final_v_v_t:")
        print(final_v_v_t)
        reconstructed_sparse_x = multiply_matrices(multiply_matrices(v_final_matrix, lambda_final_matrix),
                                                   transpose_matrix(v_final_matrix))
        print("==> reconstructed_sparse_x:")
        print(reconstructed_sparse_x)
        norm_of_difference = np.linalg.norm(x_matrix - reconstructed_sparse_x, ord="fro")
        dictionary_of_results[f"madmm_steps = {madmm_steps}, sfpca_steps = {sfpca_steps}"] = [x_matrix,
                                                                                  v_final_matrix,
                                                                                  lambda_final_matrix,
                                                                                  final_v_v_t,
                                                                                  reconstructed_sparse_x,
                                                                                  norm_of_difference]

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")

pkl.dump(dictionary_of_results, open(
    f"results_of_grid_search_{formatted_time}_lam1_{lambda_1}_lam2_{lambda_2}_m_steps_{madmm_steps}_s_steps_{sfpca_steps}.p",
    "wb"))


def grassmann_distance(X, Y):
    v, lambda_matrix, _ = np.linalg.svd(multiply_matrices(transpose_matrix(X), Y))
    lambda_matrix[lambda_matrix > 1] = 1
    lambda_matrix = np.arccos(lambda_matrix)
    return np.linalg.norm(lambda_matrix)


for key, value in dictionary_of_results.items():
    print("==================================================")
    print(f"Showing results for parameters: {key}")
    print(f"==> Showing input x_matrix:")
    print(value[0])
    print(f"==> Showing v_final_matrix:")
    print(value[1])
    print(f"==> Showing lambda_final_matrix:")
    print(value[2])
    print("==> Showing final_v_v_t:")
    print(value[3])
    print("==> Showing reconstructed x:")
    print(value[4])
    print("==> Showing norm_of_difference:")
    print(value[5])
    print("==> Showing distance between v_final_matrix and eigenvector of input_x on Grassmann manifold:")
    init_v, init_lambda, _ = np.linalg.svd(value[0])
    print(grassmann_distance(init_v, value[1]))
    print("==================================================")

print(
    f"Analysis results saved to path: results_of_grid_search_{formatted_time}_lam1_{lambda_1}_lam2_{lambda_2}_m_steps_{madmm_steps}_s_steps_{sfpca_steps}.p")
