# Basic libraries
import warnings
import numpy as np
from utils import dotdict
warnings.filterwarnings("ignore")


class MaximumSharpeRatioStrategy:
    def __init__(self):
        self.name = 'Maximum Sharpe Portfolio (MSR)'

    def generate_portfolio(self, **kwargs):
        """
        Inspired by: Eigen Portfolio Selection:
        A Robust Approach to Sharpe Ratio Maximization,
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070416
        """
        kwargs = dotdict(kwargs)
        inverse_cov_matrix = np.linalg.pinv(kwargs.cov_matrix)
        ones = np.ones(len(inverse_cov_matrix))

        numerator = np.dot(inverse_cov_matrix, kwargs.pred_returns)
        denominator = np.dot(
            np.dot(ones.transpose(), inverse_cov_matrix), kwargs.pred_returns)
        msr_portfolio_weights = numerator / denominator

        weights = {kwargs.cov_matrix.columns[i]: msr_portfolio_weights[i][0]
                   for i in range(len(msr_portfolio_weights))}

        return weights
