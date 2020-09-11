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
warnings.filterwarnings("ignore")

# Styling for plots
plt.style.use('seaborn-white')
plt.rc('grid', linestyle="dotted", color='#a0a0a0')
plt.rcParams['axes.edgecolor'] = "#04383F"


class MontoCarloSimulator:
    """
    Monto carlo simulator that calculates the historical returns distribution and uses it to predict the future returns
    """

    def __init__(self):
        print("\n--$ Simulator has been initialized")

    def price_delta(self, prices):
        """
        Percentage change
        """

        return ((prices - prices.shift()) * 100 / prices.shift())[1:]

    def draw_portfolio_performance_chart(self, returns_matrix, portfolio_weights_dictionary, strategy_name):
        """
        Draw returns chart for portfolio performance
        """

        # Get portfolio returns
        returns_matrix = np.array(returns_matrix).transpose()
        portfolio_weights_vector = np.array(
            [portfolio_weights_dictionary[symbol] for symbol in portfolio_weights_dictionary]).transpose()
        portfolio_returns = np.dot(returns_matrix, portfolio_weights_vector)
        portfolio_returns_cumulative = np.cumsum(portfolio_returns)

        # Plot
        x = np.arange(len(portfolio_returns_cumulative))
        plt.axhline(y=0, linestyle='dotted', alpha=0.3, color='black')
        plt.plot(x, portfolio_returns_cumulative, linewidth=2.0,
                 label="Projected Returns from " + str(strategy_name))
        plt.title("Simulated Future Returns", fontsize=14)
        plt.xlabel("Bars (Time Sorted)", fontsize=14)
        plt.ylabel("Cumulative Percentage Return", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    def draw_market_performance_chart(self, actual_returns, strategy_name):
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['figure.figsize'] = (12, 6)
        """
        Draw actual market returns if future data is available
        """

        # Get market returns
        cumulative_returns = np.cumsum(actual_returns)

        # Plot
        x = np.arange(len(cumulative_returns))
        plt.axhline(y=0, linestyle='dotted', alpha=0.3, color='black')
        plt.plot(x, cumulative_returns, linewidth=2.0, color='#282828',
                 linestyle='--', label="Market Index Returns")
        plt.title("Simulated Future Returns", fontsize=14)
        plt.xlabel("Bars (Time Sorted)", fontsize=14)
        plt.ylabel("Cumulative Percentage Return", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    def simulate_portfolio(self, symbol_names, portfolio_weights_dictionary, portfolio_data_dictionary, future_prices_market, test_or_predict, market_chart, strategy_name, simulation_timesteps=25):
        """
        Simulate portfolio returns in the future
        """
        returns_matrix = []

        # Iterate over each symbol to get their returns
        for symbol in symbol_names:

            # Get symbol returns using monte carlo
            historical_close_prices = portfolio_data_dictionary[symbol]["historical"]["Close"]
            future_price_predictions = self.simulate_and_get_future_prices(historical_close_prices, simulation_timesteps=max(
                simulation_timesteps, portfolio_data_dictionary[symbol]["future"].shape[0]))
            predicted_future_returns = self.price_delta(future_price_predictions)
            returns_matrix.append(predicted_future_returns)

        # Get portfolio returns
        self.draw_portfolio_performance_chart(
            returns_matrix, portfolio_weights_dictionary, strategy_name)

        # Check whether we have actual future data available or not
        if test_or_predict == 1:
            actual_future_prices_returns = self.price_delta(future_prices_market.Close)
            if market_chart:
                # Also draw the actual future returns
                self.draw_market_performance_chart(
                    actual_future_prices_returns, strategy_name)

    def simulate_and_get_future_prices(self, historical_prices, simulation_timesteps=25):

        # Get log returns from historical data
        close_prices = historical_prices
        returns = np.log((close_prices / close_prices.shift())[1:])

        # Get distribution of returns
        hist = np.histogram(returns, bins=32)
        hist_dist = scipy.stats.rv_histogram(hist)  # Distribution function

        predicted_prices = []
        # Do 25 iterations to simulate prices
        for iteration in range(25):
            new_close_prices = [close_prices.values[-1]]
            for i in range(simulation_timesteps):
                random_value = random.uniform(0, 1)
                # Get simulated return
                return_value = round(np.exp(hist_dist.ppf(random_value)), 5)
                price_next_point = new_close_prices[-1] * return_value

                # Add to list
                new_close_prices.append(price_next_point)

            predicted_prices.append(new_close_prices)

        # Calculate confidence intervals and average future returns. Conf intervals are not being used right now
        # conf_intervals = st.t.interval(0.95, len(predicted_prices), loc=np.mean(
        #     predicted_prices, axis=0), scale=st.sem(predicted_prices, axis=0))
        
        return pd.DataFrame(columns=["Close"], data = np.mean(predicted_prices, axis=0))

    def is_nan(self, object):
        """
        Check if object is null
        """
        return object != object
