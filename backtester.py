# Basic libraries
import os
import sys
import math
import scipy
import random
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from utils import dotdict
warnings.filterwarnings("ignore")


class BackTester:
    """
    Backtester module that does both backward and forward testing for our portfolios.
    """

    @classmethod
    def price_delta(self, prices):
        """
        Percentage change
        """

        return ((prices - prices.shift()) * 100 / prices.shift())[1:]

    @classmethod
    def portfolio_weight_manager(self, weight, is_long_only):
        """
        Manage portfolio weights. If portfolio is long only, 
        set the negative weights to zero.
        """
        if is_long_only == 1:
            weight = max(weight, 0)
        else:
            weight = weight
        return weight

    @classmethod
    def plot_market(self, market_returns):
        x = np.arange(len(market_returns))
        plt.plot(x, market_returns, linewidth=2.0,
                 color='#282828', label='Market Index', linestyle='--')

    @classmethod
    def plot_test(self, **kwargs):
        # Styling for plots
        kwargs = dotdict(kwargs)
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['axes.titlesize'] = "large"
        plt.rcParams['axes.labelsize'] = "medium"
        plt.rcParams['lines.linewidth'] = 2
        # Plot
        plt.axhline(y=0, linestyle='dotted', alpha=0.3, color='black')

        # Plotting styles
        kwargs.df.plot(fontsize=14, title=kwargs.title,
                       xlabel=kwargs.xlabel, ylabel=kwargs.ylabel,)

    @classmethod
    def get_historical_test(self, p_weights, data, long_only: bool):
        """
        Main backtest function. Takes in the portfolio weights and compares 
        the portfolio returns with a market index of your choice.
        """
        symbol_names = list(p_weights.keys())

        # Get invidiual returns for each stock in our portfolio
        normal_returns_matrix = []
        for symbol in symbol_names:
            symbol_historical_prices = data[symbol]["historical"]["Close"]
            symbol_historical_returns = BackTester.price_delta(
                symbol_historical_prices)
            normal_returns_matrix.append(symbol_historical_returns)

        # Get portfolio returns
        normal_returns_matrix = np.array(normal_returns_matrix).transpose()
        portfolio_weights_vector = np.array([BackTester.portfolio_weight_manager(
            p_weights[symbol], long_only) for symbol in p_weights]).transpose()
        portfolio_returns = np.dot(
            normal_returns_matrix, portfolio_weights_vector)
        return np.cumsum(portfolio_returns)

    @classmethod
    def get_market_returns(self, market_data, direction):
        assert (direction in ["historical", "future"]
                ), "direction must be 'historical' or 'future'"
        # Get future prices
        future_price_market = market_data[direction].Close
        market_returns = self.price_delta(future_price_market)
        return np.cumsum(market_returns)

    @classmethod
    def predict_future_returns(self, p_weights, data,
                               long_only: bool) -> np.ndarray:
        """
        Main future test function. If future data is available i.e is_test
        is set to 1 and future_bars set to > 0, this takes in the portfolio
        weights and compares the portfolio returns with a market index of
        your choice in the future.
        """
        symbol_names = list(p_weights.keys())
        # future_data = data_dict[symbol]["future"].Close

        # Get invidiual returns for each stock in our portfolio
        normal_returns_matrix = []
        for symbol in symbol_names:
            symbol_historical_prices = data[symbol]["future"].Close
            symbol_historical_returns = BackTester.price_delta(
                symbol_historical_prices)
            normal_returns_matrix.append(symbol_historical_returns)

        # Get portfolio returns
        normal_returns_matrix = np.array(normal_returns_matrix).transpose()
        portfolio_weights_vector = np.array([BackTester.portfolio_weight_manager(
            p_weights[symbol], long_only) for symbol in p_weights]).transpose()
        portfolio_returns = np.dot(
            normal_returns_matrix, portfolio_weights_vector)
        return np.cumsum(portfolio_returns)
