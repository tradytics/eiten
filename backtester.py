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
from utils import dotdict, get_price_deltas, get_log_returns
from utils import get_predicted_returns
warnings.filterwarnings("ignore")


class BackTester:
    """
    Backtester module that does both backward and forward testing for our portfolios.
    """

    @staticmethod
    def filter_short(weights, long_only):
        """
        Manage portfolio weights. If portfolio is long only, 
        set the negative weights to zero.
        """
        if long_only:
            return np.array([max(i, 0) for i in weights])
        return weights

    @staticmethod
    def plot_market(market_returns):
        x = np.arange(len(market_returns))
        plt.plot(x, market_returns, linewidth=2.0,
                 color='#282828', label='Market Index', linestyle='--')

    @staticmethod
    def plot_test(**kwargs):
        # Styling for plots
        kwargs = dotdict(kwargs)
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['figure.figsize'] = (18, 6)
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['axes.titlesize'] = "large"
        plt.rcParams['axes.labelsize'] = "medium"
        plt.rcParams['lines.linewidth'] = 2
        # Plot
        plt.axhline(y=0, linestyle='dotted', alpha=0.3, color='black')

        # Plotting styles
        kwargs.df.plot(fontsize=14, title=kwargs.title,
                       xlabel=kwargs.xlabel, ylabel=kwargs.ylabel,)

    @staticmethod
    def get_test(p_weights, data, direction: str, long_only: bool):
        """
        Main backtest function. Takes in the portfolio weights and compares
        the portfolio returns with a market index of your choice.
        """

        assert (direction in ["historical", "future", "sim"]
                ), "direction must be 'historical', 'future' or 'sim'"

        # Get invidiual returns for each stock in our portfolio
        normal_returns_matrix = []
        symbol_historical_prices = data[direction]

        # Get portfolio returns
        normal_returns_matrix = get_price_deltas(
            symbol_historical_prices).cumsum()
        portfolio_returns = np.dot(normal_returns_matrix, p_weights)

        return portfolio_returns

    @staticmethod
    def get_market_returns(market_data, direction):
        assert (direction in ["historical", "future", "sim"]
                ), "direction must be 'historical', 'future' or 'sim'"
        # Get future prices
        future_price_market = market_data[direction]
        market_returns = get_price_deltas(future_price_market)
        return np.cumsum(market_returns)

    @staticmethod
    def simulate_future_prices(data: dict, r_func,
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
        returns = r_func(close_prices)
        symbol_simulations = []
        for col in returns.columns:
            # Get distribution of returns
            hist = np.histogram(returns[col], bins=32)
            hist_dist = st.rv_histogram(hist)  # Distribution function

            simulations = []
            # Do 25 iterations to simulate prices
            for _ in range(100):
                timeseries = [close_prices[col].values[-1]]
                for _ in range(min(simulation_timesteps,
                                   data["future"].shape[0])):
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
