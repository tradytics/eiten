# Basic libraries
import warnings
import numpy as np
from utils import dotdict, normalize_weights
warnings.filterwarnings("ignore")


class MinimumVarianceStrategy:
    def __init__(self):
        self.name = "Minimum Variance Portfolio (MVP)"

    def generate_portfolio(self, **kwargs):
        """
        Inspired by: https://srome.github.io/Eigenvesting-II-Optimize-Your-Portfolio-With-Optimization/
        """
        kwargs = dotdict(kwargs)

        inverse_cov_matrix = np.linalg.pinv(kwargs.cov_matrix)
        ones = np.ones(len(inverse_cov_matrix))
        inverse_dot_ones = np.dot(inverse_cov_matrix, ones)
        min_var_weights = inverse_dot_ones / np.dot(inverse_dot_ones, ones)
        min_var_weights = normalize_weights(min_var_weights)
        weights = {kwargs.cov_matrix.columns[i]: min_var_weights[i]
                   for i in range(min_var_weights.shape[0])}
        return weights
