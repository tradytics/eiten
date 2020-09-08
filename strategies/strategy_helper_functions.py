# Basic libraries
import os
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class StrategyHelperFunctions:
	def __init__(self):
		print("Helper functions have been created")
		
	def random_matrix_theory_based_cov(self, returns_matrix):
		"""
		This is inspired by the excellent post @ https://srome.github.io/Eigenvesting-III-Random-Matrix-Filtering-In-Finance/
		"""

		# Calculate variance and std, will come in handy during reconstruction
		variances = np.diag(np.cov(returns_matrix))
		standard_deviations = np.sqrt(variances)

		# Get correlation matrix and compute eigen vectors and values
		correlation_matrix = np.corrcoef(returns_matrix)
		eig_values, eig_vectors = np.linalg.eigh(correlation_matrix)
		
		# Get maximum theoretical eigen value for a random matrix
		sigma = 1 # The variance for all of the standardized log returns is 1
		Q = len(returns_matrix[0]) / len(returns_matrix)
		max_theoretical_eval = np.power(sigma*(1 + np.sqrt(1/Q)),2)
		
		# Prune random eigen values
		eig_values_pruned = eig_values[eig_values > max_theoretical_eval]
		eig_values[eig_values <= max_theoretical_eval] = 0

		# Reconstruct the covariance matrix from the correlation matrix and filtered eigen values
		temp = np.dot(eig_vectors, np.dot(np.diag(eig_values), np.transpose(eig_vectors)))
		np.fill_diagonal(temp, 1)
		filtered_matrix = temp
		filtered_cov_matrix = np.dot(np.diag(standard_deviations), 
                               np.dot(filtered_matrix,np.diag(standard_deviations)))
		
		return filtered_cov_matrix