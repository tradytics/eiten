# Basic libraries
import os
import warnings
import numpy as np
from utils import dotdict
warnings.filterwarnings("ignore")


class EigenPortfolioStrategy:
    def __init__(self):
        self.name = "Eigen Portfolio"

    def generate_portfolio(self, **kwargs):
        """
        Inspired by: https://srome.github.io/Eigenvesting-I-Linear-Algebra-Can-Help-You-Choose-Your-Stock-Portfolio/
        """
        kwargs = dotdict(kwargs)
        eigh_values, eigh_vectors = np.linalg.eigh(kwargs.cov_matrix)
        # We don't need this but in case someone wants to analyze
        # market_eigen_portfolio = eig_vectors[:, -1] / np.sum(eig_vectors[:, -1])
        # This is a portfolio that is uncorrelated to market and still yields good returns
        eigen_portfolio = eigh_vectors[:, -kwargs.p_number] / \
            np.sum(eigh_vectors[:, -kwargs.p_number])

        weights = {kwargs.cov_matrix.columns[i]: eigen_portfolio[i]
                   for i in range(eigen_portfolio.shape[0])}
        return weights

