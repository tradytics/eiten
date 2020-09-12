# Basic libraries
import warnings
import numpy as np
warnings.filterwarnings("ignore")


class MaximumSharpeRatioStrategy:
    def __init__(self):
        print("Maximum sharpe ratio strategy has been created")

    def generate_portfolio(self, cov_matrix, returns_vector):
        """
        Inspired by: Eigen Portfolio Selection:
        A Robust Approach to Sharpe Ratio Maximization,
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070416
        """
        inverse_cov_matrix = np.linalg.pinv(cov_matrix)
        ones = np.ones(len(inverse_cov_matrix))

        numerator = np.dot(inverse_cov_matrix, returns_vector)
        denominator = np.dot(
            np.dot(ones.transpose(), inverse_cov_matrix), returns_vector)
        msr_portfolio_weights = numerator / denominator

        weights = {cov_matrix.columns[i]: msr_portfolio_weights[i][0]
                   for i in range(len(msr_portfolio_weights))}

        return weights
