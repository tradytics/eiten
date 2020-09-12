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
    def plot_market(self, **kwargs):
        kwargs = dotdict(kwargs)
        x = np.arange(len(kwargs.market_returns_cumulative))
        plt.plot(x, kwargs.market_returns_cumulative, linewidth=2.0,
                 color='#282828', label='Market Index', linestyle='--')

    @classmethod
    def plot_test(self, **kwargs):
        # Styling for plots
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        # Plot
        x = np.arange(len(kwargs.portfolio_returns_cumulative))
        plt.axhline(y=0, linestyle='dotted', alpha=0.3, color='black')
        plt.plot(x, kwargs.portfolio_returns_cumulative,
                 linewidth=2.0, label=kwargs.strategy_name)

        # Plotting styles
        plt.title(kwargs.title, fontsize=14)
        plt.xlabel(kwargs.xlabel, fontsize=14)
        plt.ylabel(kwargs.ylabel, fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

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

        # # Plot returns
        # x = np.arange(len(portfolio_returns_cumulative))
        # plt.plot(x, portfolio_returns_cumulative,
        #          linewidth=2.0, label=strategy_name)
        # plt.axhline(y=0, linestyle='dotted', alpha=0.3, color='black')

        # # Plotting styles
        # plt.title("Backtest Results", fontsize=14)
        # plt.xlabel("Bars (Time Sorted)", fontsize=14)
        # plt.ylabel("Cumulative Percentage Return", fontsize=14)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)

    @classmethod
    def get_future_market_returns(self, market_data):
        # Get future prices
        future_price_market = market_data["future"].Close
        market_returns = self.price_delta(future_price_market)
        return np.cumsum(market_returns)

    def get_historical_market_returns(self, historical_data):
        # Get market returns during the backtesting time
        historical_prices = historical_data["historical"]["Close"]
        market_returns = self.price_delta(historical_prices)
        return np.cumsum(market_returns)

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
