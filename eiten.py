import math
import numpy as np
import matplotlib.pyplot as plt

# Load our modules
from data_loader import DataEngine
from simulator import MontoCarloSimulator
from backtester import BackTester
from strategy_manager import StrategyManager


class Eiten:
    def __init__(self, args):
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['figure.figsize'] = (12, 6)

        print("\n--* Eiten has been initialized...")
        self.args = args

        # Create data engine
        self.dataEngine = DataEngine(args)

        # Monto carlo simulator
        self.simulator = MontoCarloSimulator()

        # Strategy manager
        self.strategyManager = StrategyManager()

        # Back tester
        self.backTester = BackTester()

        # Data dictionary
        self.data_dictionary = {}

        print('\n')

    def calculate_percentage_change(self, old, new):
        """
        Calculate percentage change
        """
        return ((new - old) * 100) / old

    def create_returns(self, historical_price_info):
        """
        Create log return matrix, percentage return matrix, and mean return 
        vector
        """

        returns_matrix = []
        returns_matrix_percentages = []
        predicted_return_vectors = []
        for i in range(0, len(historical_price_info)):
            close_prices = list(historical_price_info[i]["Close"])
            log_returns = [math.log(close_prices[i] / close_prices[i - 1])
                           for i in range(1, len(close_prices))]
            percentage_returns = [self.calculate_percentage_change(
                close_prices[i - 1], close_prices[i]) for i in range(1, len(close_prices))]

            total_data = len(close_prices)

            # Expected returns in future. We can either use historical returns as future returns on try to simulate future returns and take the mean. For simulation, you can modify the functions in simulator to use here.
            future_expected_returns = np.mean([(self.calculate_percentage_change(close_prices[i - 1], close_prices[i])) / (
                total_data - i) for i in range(1, len(close_prices))])  # More focus on recent returns

            # Add to matrices
            returns_matrix.append(log_returns)
            returns_matrix_percentages.append(percentage_returns)

            # Add returns to vector
            # Assuming that future returns are similar to past returns
            predicted_return_vectors.append(future_expected_returns)

        # Convert to numpy arrays for one liner calculations
        predicted_return_vectors = np.array(predicted_return_vectors)
        returns_matrix = np.array(returns_matrix)
        returns_matrix_percentages = np.array(returns_matrix_percentages)

        return predicted_return_vectors, returns_matrix, returns_matrix_percentages

    def load_data(self):
        """
        Loads data needed for analysis
        """
        # Gather data for all stocks in a dictionary format
        # Dictionary keys will be -> historical_prices, future_prices
        self.data_dictionary = self.dataEngine.collect_data_for_all_tickers()

        # Add data to lists
        symbol_names = list(sorted(self.data_dictionary.keys()))
        historical_price_info, future_prices = [], []
        for symbol in symbol_names:
            historical_price_info.append(
                self.data_dictionary[symbol]["historical_prices"])
            future_prices.append(self.data_dictionary[symbol]["future_prices"])

        # Get return matrices and vectors
        predicted_return_vectors, returns_matrix, returns_matrix_percentages = self.create_returns(
            historical_price_info)
        return historical_price_info, future_prices, symbol_names, predicted_return_vectors, returns_matrix, returns_matrix_percentages

    def run_strategies(self):
        """
        Run strategies, back and future test them, and simulate the returns.
        """
        historical_price_info, future_prices, symbol_names, predicted_return_vectors, returns_matrix, returns_matrix_percentages = self.load_data()
        historical_price_market, future_prices_market = self.dataEngine.get_market_index_price()

        # Calculate covariance matrix
        covariance_matrix = np.cov(returns_matrix)

        # Use random matrix theory to filter out the noisy eigen values
        if self.args.apply_noise_filtering:
            print(
                "\n** Applying random matrix theory to filter out noise in the covariance matrix...\n")
            covariance_matrix = self.strategyManager.random_matrix_theory_based_cov(
                returns_matrix)

        # Get weights for the portfolio
        eigen_portfolio_weights_dictionary = self.strategyManager.calculate_eigen_portfolio(
            symbol_names, covariance_matrix, self.args.eigen_portfolio_number)
        mvp_portfolio_weights_dictionary = self.strategyManager.calculate_minimum_variance_portfolio(
            symbol_names, covariance_matrix)
        msr_portfolio_weights_dictionary = self.strategyManager.calculate_maximum_sharpe_portfolio(
            symbol_names, covariance_matrix, predicted_return_vectors)
        ga_portfolio_weights_dictionary = self.strategyManager.calculate_genetic_algo_portfolio(
            symbol_names, returns_matrix_percentages)

        # Print weights
        print("\n*% Printing portfolio weights...")
        self.print_and_plot_portfolio_weights(
            eigen_portfolio_weights_dictionary, 'Eigen Portfolio', plot_num=1)
        self.print_and_plot_portfolio_weights(
            mvp_portfolio_weights_dictionary, 'Minimum Variance Portfolio (MVP)', plot_num=2)
        self.print_and_plot_portfolio_weights(
            msr_portfolio_weights_dictionary, 'Maximum Sharpe Portfolio (MSR)', plot_num=3)
        self.print_and_plot_portfolio_weights(
            ga_portfolio_weights_dictionary, 'Genetic Algo (GA)', plot_num=4)
        self.draw_plot("output/weights.png")

        # Back test
        print("\n*& Backtesting the portfolios...")
        self.backTester.back_test(symbol_names, eigen_portfolio_weights_dictionary,
                                  self.data_dictionary,
                                  historical_price_market,
                                  self.args.only_long,
                                  market_chart=True,
                                  strategy_name='Eigen Portfolio')
        self.backTester.back_test(symbol_names,
                                  mvp_portfolio_weights_dictionary,
                                  self.data_dictionary, historical_price_market,
                                  self.args.only_long,
                                  market_chart=False,
                                  strategy_name='Minimum Variance Portfolio (MVP)')
        self.backTester.back_test(symbol_names, msr_portfolio_weights_dictionary,
                                  self.data_dictionary,
                                  historical_price_market,
                                  self.args.only_long,
                                  market_chart=False,
                                  strategy_name='Maximum Sharpe Portfolio (MSR)')
        self.backTester.back_test(symbol_names,
                                  ga_portfolio_weights_dictionary,
                                  self.data_dictionary,
                                  historical_price_market,
                                  self.args.only_long,
                                  market_chart=False,
                                  strategy_name='Genetic Algo (GA)')
        self.draw_plot("output/backtest.png")

        if self.args.is_test:
            print("\n#^ Future testing the portfolios...")
            # Future test
            self.backTester.future_test(symbol_names,
                                        eigen_portfolio_weights_dictionary,
                                        self.data_dictionary,
                                        future_prices_market,
                                        self.args.only_long,
                                        market_chart=True,
                                        strategy_name='Eigen Portfolio')
            self.backTester.future_test(symbol_names,
                                        mvp_portfolio_weights_dictionary,
                                        self.data_dictionary,
                                        future_prices_market,
                                        self.args.only_long,
                                        market_chart=False,
                                        strategy_name='Minimum Variance Portfolio (MVP)')
            self.backTester.future_test(symbol_names,
                                        msr_portfolio_weights_dictionary,
                                        self.data_dictionary,
                                        future_prices_market,
                                        self.args.only_long,
                                        market_chart=False,
                                        strategy_name='Maximum Sharpe Portfolio (MSR)')
            self.backTester.future_test(symbol_names,
                                        ga_portfolio_weights_dictionary,
                                        self.data_dictionary,
                                        future_prices_market,
                                        self.args.only_long,
                                        market_chart=False,
                                        strategy_name='Genetic Algo (GA)')
            self.draw_plot("output/future_tests.png")

        # Simulation
        print("\n+$ Simulating future prices using monte carlo...")
        self.simulator.simulate_portfolio(symbol_names,
                                          eigen_portfolio_weights_dictionary,
                                          self.data_dictionary,
                                          future_prices_market,
                                          self.args.is_test,
                                          market_chart=True,
                                          strategy_name='Eigen Portfolio')
        self.simulator.simulate_portfolio(symbol_names,
                                          eigen_portfolio_weights_dictionary,
                                          self.data_dictionary,
                                          future_prices_market,
                                          self.args.is_test,
                                          market_chart=False,
                                          strategy_name='Minimum Variance Portfolio (MVP)')
        self.simulator.simulate_portfolio(symbol_names,
                                          eigen_portfolio_weights_dictionary,
                                          self.data_dictionary,
                                          future_prices_market,
                                          self.args.is_test,
                                          market_chart=False,
                                          strategy_name='Maximum Sharpe Portfolio (MSR)')
        self.simulator.simulate_portfolio(symbol_names,
                                          ga_portfolio_weights_dictionary,
                                          self.data_dictionary,
                                          future_prices_market,
                                          self.args.is_test,
                                          market_chart=False,
                                          strategy_name='Genetic Algo (GA)')
        self.draw_plot("output/monte_carlo.png")

    def draw_plot(self, filename="output/graph.png"):
        """
        Draw plots
        """
        # Styling for plots

        plt.grid()
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()
        
        """if self.args.save_plot:
            plt.savefig(filename)
        else:
            plt.tight_layout()
            plt.show()""" # Plots were not being generated properly. Need to fix this.

    def print_and_plot_portfolio_weights(self, weights_dictionary: dict, strategy, plot_num: int) -> None:
        print("\n-------- Weights for %s --------" % strategy)
        symbols = list(sorted(weights_dictionary.keys()))
        symbol_weights = []
        for symbol in symbols:
            print("Symbol: %s, Weight: %.4f" %
                  (symbol, weights_dictionary[symbol]))
            symbol_weights.append(weights_dictionary[symbol])

        # Plot
        width = 0.1
        x = np.arange(len(symbol_weights))
        plt.bar(x + (width * (plot_num - 1)) + 0.05,
                symbol_weights, label=strategy, width=width)
        plt.xticks(x, symbols, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Symbols", fontsize=14)
        plt.ylabel("Weight in Portfolio", fontsize=14)
        plt.title("Portfolio Weights for Different Strategies", fontsize=14)
