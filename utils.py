# Basic libraries
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def normalize_weights(w):
    pos_sum = 0
    neg_sum = 0
    for i in w:
        if i > 0:
            pos_sum += i
        else:
            neg_sum += i
    neg_sum = abs(neg_sum)
    for i in range(len(w)):
        if w[i] > 0:
            w[i] /= pos_sum
        else:
            w[i] /= neg_sum
    return w



def random_matrix_theory_based_cov(log_returns):
    """
    This is inspired by the excellent post @
    https://srome.github.io/Eigenvesting-III-Random-Matrix-Filtering-In-Finance/
    """
    returns_matrix = log_returns.T

    # Calculate variance and std, will come in handy during reconstruction
    variances = np.diag(np.cov(returns_matrix))
    standard_deviations = np.sqrt(variances)

    # Get correlation matrix and compute eigen vectors and values
    correlation_matrix = np.corrcoef(returns_matrix)
    eig_values, eig_vectors = np.linalg.eigh(correlation_matrix)

    # Get maximum theoretical eigen value for a random matrix
    sigma = 1  # The variance for all of the standardized log returns is 1
    Q = returns_matrix.shape[1] / returns_matrix.shape[0]
    max_theoretical_eval = np.power(sigma*(1 + np.sqrt(1/Q)), 2)

    # Prune random eigen values
    # eig_values_pruned = eig_values[eig_values > max_theoretical_eval]
    eig_values[eig_values <= max_theoretical_eval] = 0

    # Reconstruct the covariance matrix from the correlation matrix
    # and filtered eigen values
    temp = np.dot(eig_vectors, np.dot(
        np.diag(eig_values), np.transpose(eig_vectors)))
    np.fill_diagonal(temp, 1)
    filtered_matrix = temp
    filtered_cov_matrix = np.dot(np.diag(standard_deviations),
                                 np.dot(filtered_matrix,
                                        np.diag(standard_deviations)))
    return pd.DataFrame(columns=log_returns.columns,
                        data=filtered_cov_matrix)


def get_price_deltas(prices: pd.DataFrame):
    """
    Calculate ratio of change
    """
    return ((prices - prices.shift()) / prices.shift())[1:]

def get_capm_returns(data:pd.DataFrame) -> pd.DataFrame:
    #not correct
    return data.std() * (get_price_deltas(data).mean() )

def get_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    return np.log((data / data.shift())[1:])


def get_exp_returns(data: pd.DataFrame) -> pd.DataFrame:
    return get_price_deltas(data).ewm(span=len(data)).mean()


def get_predicted_returns(data: pd.DataFrame) -> pd.DataFrame:
    return get_price_deltas(data).div(
        np.array(np.arange((len(data) - 1), 0, -1)), axis=0)
