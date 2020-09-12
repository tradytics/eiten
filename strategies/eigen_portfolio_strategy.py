# Basic libraries
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")


class EigenPortfolioStrategy:
    def __init__(self):
        print("Eigen portfolio strategy has been created")

    def generate_portfolio(self, cov_matrix, eigen_portfolio_number):
        """
        Inspired by: https://srome.github.io/Eigenvesting-I-Linear-Algebra-Can-Help-You-Choose-Your-Stock-Portfolio/
        """
        eigh_values, eigh_vectors = np.linalg.eigh(cov_matrix)
        print("eig_values", eigh_values.shape, "eig_vectors", eigh_vectors.shape)
        # We don't need this but in case someone wants to analyze
        # market_eigen_portfolio = eig_vectors[:, -1] / np.sum(eig_vectors[:, -1])
        # This is a portfolio that is uncorrelated to market and still yields good returns
        eigen_portfolio = eigh_vectors[:, -eigen_portfolio_number] / \
            np.sum(eigh_vectors[:, -eigen_portfolio_number])


        weights = {cov_matrix.columns[i]: eigen_portfolio[i]
                   for i in range(eigen_portfolio.shape[0])}
        return weights
