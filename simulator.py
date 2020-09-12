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

	def calculate_percentage_change(self, old, new):
		"""
		Percentage change
		"""
		return ((new - old) * 100) / old

	def draw_portfolio_performance_chart(self, returns_matrix, portfolio_weights_dictionary, strategy_name): 
		"""
		Draw returns chart for portfolio performance
		"""

		# Get portfolio returns
		returns_matrix = np.array(returns_matrix).transpose()
		portfolio_weights_vector = np.array([portfolio_weights_dictionary[symbol] for symbol in portfolio_weights_dictionary]).transpose()
		portfolio_returns = np.dot(returns_matrix, portfolio_weights_vector)
		portfolio_returns_cumulative = np.cumsum(portfolio_returns)

		# Plot
		x = np.arange(len(portfolio_returns_cumulative))
		plt.axhline(y = 0, linestyle = 'dotted', alpha = 0.3, color = 'black')
		plt.plot(x, portfolio_returns_cumulative, linewidth = 2.0, label = "Projected Returns from " + str(strategy_name))
		plt.title("Simulated Future Returns", fontsize = 14)
		plt.xlabel("Bars (Time Sorted)", fontsize = 14)
		plt.ylabel("Cumulative Percentage Return", fontsize = 14)
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)

	def draw_market_performance_chart(self, actual_returns, strategy_name):
		"""
		Draw actual market returns if future data is available
		"""

		# Get market returns
		cumulative_returns = np.cumsum(actual_returns)

		# Plot
		x = np.arange(len(cumulative_returns))
		plt.axhline(y = 0, linestyle = 'dotted', alpha = 0.3, color = 'black')
		plt.plot(x, cumulative_returns, linewidth = 2.0, color = '#282828', linestyle = '--', label = "Market Index Returns")
		plt.title("Simulated Future Returns", fontsize = 14)
		plt.xlabel("Bars (Time Sorted)", fontsize = 14)
		plt.ylabel("Cumulative Percentage Return", fontsize = 14)
		plt.xticks(fontsize = 14)
		plt.yticks(fontsize = 14)

	def simulate_portfolio(self, symbol_names, portfolio_weights_dictionary, portfolio_data_dictionary, future_prices_market, test_or_predict, market_chart, strategy_name, simulation_timesteps = 25):
		"""
		Simulate portfolio returns in the future
		"""
		returns_matrix = []
		actual_returns_matrix = []

		# Iterate over each symbol to get their returns
		for symbol in symbol_names:

			# Get symbol returns using monte carlo
			historical_close_prices = list(portfolio_data_dictionary[symbol]["historical_prices"]["Close"])
			future_price_predictions, _ = self.simulate_and_get_future_prices(historical_close_prices, simulation_timesteps = max(simulation_timesteps, len(list(portfolio_data_dictionary[symbol]["future_prices"]))))
			predicted_future_returns = [self.calculate_percentage_change(future_price_predictions[i - 1], future_price_predictions[i]) for i in range(1, len(future_price_predictions))]
			returns_matrix.append(predicted_future_returns)
			
			
		# Get portfolio returns
		self.draw_portfolio_performance_chart(returns_matrix, portfolio_weights_dictionary, strategy_name)

		# Check whether we have actual future data available or not
		if test_or_predict == 1:
			future_prices_market = [item[4] for item in list(future_prices_market)]
			actual_future_prices_returns = [self.calculate_percentage_change(future_prices_market[i - 1], future_prices_market[i]) for i in range(1, len(future_prices_market))]
			if market_chart == True:
				# Also draw the actual future returns
				self.draw_market_performance_chart(actual_future_prices_returns, strategy_name)
		
	def simulate_and_get_future_prices(self, historical_prices, simulation_timesteps = 25):

		# Get log returns from historical data
		close_prices = historical_prices
		returns = [math.log(close_prices[i] / close_prices[i - 1]) for i in range(1, len(close_prices))]

		# Get distribution of returns
		hist = np.histogram(returns, bins = 32)
		hist_dist = scipy.stats.rv_histogram(hist) # Distribution function

		predicted_prices = []
		# Do 25 iterations to simulate prices
		for iteration in range(25):
			new_close_prices = [close_prices[-1]]
			new_close_prices_percentages = []
			for i in range(simulation_timesteps):
				random_value = random.uniform(0, 1)
				return_value = round(np.exp(hist_dist.ppf(random_value)), 5) # Get simulated return
				price_last_point = new_close_prices[-1]
				price_next_point = price_last_point * return_value
				percentage_change = self.calculate_percentage_change(price_last_point, price_next_point)
					
				# Add to list
				new_close_prices.append(price_next_point)
					
			predicted_prices.append(new_close_prices)

		# Calculate confidence intervals and average future returns. Conf intervals are not being used right now
		conf_intervals = st.t.interval(0.95, len(predicted_prices), loc=np.mean(predicted_prices, axis = 0), scale=st.sem(predicted_prices, axis = 0))
		predicted_prices_mean = np.mean(predicted_prices, axis = 0)
		return predicted_prices_mean, conf_intervals

	def is_nan(self, object):
		"""
		Check if object is null
		"""
		return object != object

		
