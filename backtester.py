# Basic libraries
import os
import sys
import math
import scipy
import random
import collections
import numpy as np
import pandas as pd
import scipy.stats as st
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
    def filter_short(self, weights, long_only):
        """
        Manage portfolio weights. If portfolio is long only, 
        set the negative weights to zero.
        """
        if long_only:
            return np.array([max(i, 0) for i in weights])
        return weights

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
    def get_test(self, p_weights, data, direction: str, long_only: bool):
        """
        Main backtest function. Takes in the portfolio weights and compares 
        the portfolio returns with a market index of your choice.
        """

        assert (direction in ["historical", "future", "sim"]
                ), "direction must be 'historical', 'future' or 'sim'"

        # Get invidiual returns for each stock in our portfolio
        normal_returns_matrix = []
        symbol_historical_prices = data[direction]
        symbol_historical_returns = BackTester.price_delta(
            symbol_historical_prices)
        normal_returns_matrix.append(symbol_historical_returns)

        # Get portfolio returns
        normal_returns_matrix = np.array(normal_returns_matrix)
        portfolio_returns = np.dot(
            normal_returns_matrix, list(p_weights.values()))
        return np.cumsum(portfolio_returns)

    @classmethod
    def get_market_returns(self, market_data, direction):
        assert (direction in ["historical", "future", "sim"]
                ), "direction must be 'historical', 'future' or 'sim'"
        # Get future prices
        future_price_market = market_data[direction]
        market_returns = self.price_delta(future_price_market)
        return np.cumsum(market_returns)

    @classmethod
    def simulate_future_prices(self, data: dict,
                               simulation_timesteps: int = 30) ->pd.DataFrame:
        """Simulates the price of a collection of stocks in the future

        [description]
        :param data: a dictionary with the loaded data
        :type data: dict
        :param simulation_timesteps: number of steps, defaults to 30
        :type simulation_timesteps: number, optional
        :returns: A dataframe of simulated prices
        :rtype: pd.Dataframe
        """
        

        # Get log returns from historical data
        close_prices = data["historical"]
        returns = np.log((close_prices / close_prices.shift())[1:])
        symbol_simulations = []
        for col in returns.columns:
            # Get distribution of returns
            hist = np.histogram(returns[col], bins=32)
            hist_dist = st.rv_histogram(hist)  # Distribution function

            simulations = []
            # Do 25 iterations to simulate prices
            for iteration in range(25):
                timeseries = [close_prices[col].values[-1]]
                for i in range(simulation_timesteps):
                    # Get simulated return
                    return_value = np.round(
                        np.exp(hist_dist.ppf(random.uniform(0, 1))), 5)
                    data_point = timeseries[-1] * return_value

                    # Add to list
                    timeseries.append(data_point)
                # print(timeseries)
                simulations.append(np.array(timeseries))
            symbol_simulations.append(np.mean(np.array(simulations), axis=0))
        
        df = pd.DataFrame(columns=returns.columns,
                          data=np.array(symbol_simulations).T)
        return df
