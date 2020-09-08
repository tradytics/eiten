# Basic libraries
import os
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class MinimumVarianceStrategy:
	def __init__(self):
		print("Minimum Variance strategy has been created")
		
	def generate_portfolio(self, symbols, covariance_matrix):
		"""
		Inspired by: https://srome.github.io/Eigenvesting-II-Optimize-Your-Portfolio-With-Optimization/
		"""
		inverse_cov_matrix = np.linalg.pinv(covariance_matrix)
		ones = np.ones(len(inverse_cov_matrix))
		inverse_dot_ones = np.dot(inverse_cov_matrix, ones)
		min_var_weights = inverse_dot_ones / np.dot( inverse_dot_ones, ones)
		portfolio_weights_dictionary = dict([(symbols[x], min_var_weights[x]) for x in range(0, len(min_var_weights))])
		return portfolio_weights_dictionary
