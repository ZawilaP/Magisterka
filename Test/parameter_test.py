import numpy as np
import pickle as pkl
import datetime

from ManifoldOptimization.SFPCA.sfpca import SFPCA
from ManifoldOptimization.Utils.matrix_operations import multiply_matrices, transpose_matrix

x_matrix = np.array([[1, 0.82, 0.82, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.82, 1, 0.82, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.82, 0.82, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 1, 0.92, 0.92, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.92, 1, 0.92, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.92, 0.92, 1, 0.001, 0.001, 0.001],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1, 0.44, 0.44],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.44, 1, 0.44],
                     [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.44, 0.44, 1]])

rho = 1
madmm_steps = 5
sfpca_steps = 10
dictionary_of_results = {}
lambda_1_range = [0.01, 0.001, 0.0015]
lambda_2_range = [0.1, 0.5, 0.15, 0.1]

for lambda_1 in lambda_1_range:
    for lambda_2 in lambda_2_range:
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
        dictionary_of_results[f"lambda_1 = {lambda_1}, lambda_2 = {lambda_2}"] = [v_final_matrix,
                                                                                  lambda_final_matrix,
                                                                                  final_v_v_t,
                                                                                  reconstructed_sparse_x,
                                                                                  norm_of_difference]

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")

pkl.dump(dictionary_of_results, open(f"results_of_grid_search_{formatted_time}_lam1_{lambda_1_range}_lam2_{lambda_2_range}.p", "wb"))

for key, value in dictionary_of_results.items():
    print("==================================================")
    print(f"Showing results for parameters: {key}")
    print(f"==> Showing v_final_matrix:")
    print(value[0])
    print(f"==> Showing lambda_final_matrix:")
    print(value[1])
    print("==> Showing final_v_v_t:")
    print(value[2])
    print("==> Showing reconstructed x:")
    print(value[3])
    print("==> Showing norm_of_difference:")
    print(value[4])
    print("==================================================")