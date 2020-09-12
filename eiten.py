import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load our modules
from data_loader import DataEngine
from simulator import MontoCarloSimulator
from backtester import BackTester
from utils import random_matrix_theory_based_cov
from utils import dotdict
from strategies import portfolios


class Eiten:
    def __init__(self, args: dict = None):
        if args is None:
            arg_types = {"str": str, "int": int, "bool": bool}
            x = json.load(open("commands.json", "r"))
            args = dotdict(
                {i["comm"][2:]: arg_types[i["type"]](i["default"]) for i in x})

        print("\n--* Eiten has been initialized...")
        self.args = args

        # Create data engine
        self.dataEngine = DataEngine(args)

        # Monte carlo simulator
        self.simulator = MontoCarloSimulator()

        # Back tester
        self.backTester = BackTester()

        # Data dictionary
        self.data_dict = {}  # {"market": args.market_index}
        self.market_data = {}

        print('\n')

    def _get_price_delta(self, prices: pd.DataFrame):
        """
        Calculate percentage change
        """
        return ((prices - prices.shift()) * 100 / prices.shift())[1:]

    def _get_abstract_returns(self, f):
        """Abstract form of getting returns

        This function takes a function f as parameter and applies the function
        to the closing prices of the current data dictionary

        :param f: a function that takes a dictionary and returns some returns
        :type f: function
        :returns: A Pandas Dataframe of returns for each asset
        :rtype: pd.Dataframe
        """
        res = pd.DataFrame(pd.DataFrame(columns=list(self.data_dict.keys())))
        for i in self.data_dict:
            c = self.data_dict[i]["historical"].Close
            res[i] = f(c)
        return res

    def _get_perc_returns(self):
        return self._get_abstract_returns(self._get_price_delta)

    def _get_log_return(self, data: pd.DataFrame) -> np.ndarray:
        return np.log((data / data.shift())[1:])

    def _get_log_returns(self):
        return self._get_abstract_returns(self._get_log_return)

    def _get_predicted_return(self, data: pd.DataFrame) -> np.ndarray:
        return np.array([np.mean(self._get_price_delta(data) /
                                 np.array(np.arange(len(data) - 1, 0, -1)))])

    def _get_predicted_returns(self):
        return self._get_abstract_returns(self._get_predicted_return)

    def load_data(self):
        """
        Loads data needed for analysis
        """
        # Gather data for all stocks in a dictionary format
        # Dictionary keys will be -> historical, future
        self.data_dict = self.dataEngine.collect_data_for_all_tickers()
        p, f = self.dataEngine.get_data(self.args.market_index)
        self.market_data["historical"], self.market_data["future"] = p, f
        # Get return matrices and vectors
        return self.data_dict

    def run_strategies(self):
        """
        Run strategies, back and future test them, and simulate the returns.
        """
        self.load_data()

        # Calculate covariance matrix
        log_returns = self._get_log_returns()
        cov_matrix = log_returns.cov()

        # Use random matrix theory to filter out the noisy eigen values
        if self.args.apply_noise_filtering:
            print(
                "\n** Applying random matrix theory to filter out noise in the covariance matrix...\n")
            cov_matrix = random_matrix_theory_based_cov(log_returns)

        pred_returns = self._get_predicted_returns()
        perc_returns = self._get_perc_returns()

        self.portfolios = {}
        # Get weights for the portfolio
        for p in portfolios:
            name = p.name
            weights = p.generate_portfolio(
                cov_matrix=cov_matrix, p_number=self.args.eigen_portfolio_number,
                pred_returns=pred_returns.T,
                perc_returns=perc_returns)
            self.portfolios[name] = weights

        # Print weights
        print("\n*% Printing portfolio weights...")
        p_count = 1
        for i in self.portfolios:
            self.print_and_plot_portfolio_weights(
                self.portfolios[i], i, plot_num=p_count)
            p_count += 1
        self.draw_plot("output/weights.png")

        # Back test
        print("\n*& Backtesting the portfolios...")

        for i in self.portfolios:

            self.backTester.back_test(self.portfolios[i],
                                      self.data_dict,
                                      self.market_data,
                                      self.args.only_long,
                                      market_chart=True,
                                      strategy_name=i)
        self.draw_plot("output/backtest.png")

        if self.args.is_test:
            print("\n#^ Future testing the portfolios...")
            # Future test
            for i in self.portfolios:

                self.backTester.future_test(self.portfolios[i],
                                            self.data_dict,
                                            self.market_data,
                                            self.args.only_long,
                                            market_chart=True,
                                            strategy_name=i)

            self.draw_plot("output/future_tests.png")

        # Simulation
        print("\n+$ Simulating future prices using monte carlo...")
        for i in self.portfolios:
            self.simulator.simulate_portfolio(self.portfolios[i],
                                              self.data_dict,
                                              self.market_data,
                                              self.args.is_test,
                                              market_chart=True,
                                              strategy_name=i)
        self.draw_plot("output/monte_carlo.png")

    def draw_plot(self, filename="output/graph.png"):
        """
        Draw plots
        """
        # Styling for plots
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['figure.figsize'] = (12, 6)

        plt.grid()
        plt.legend(fontsize=14)
        if self.args.save_plot:
            plt.savefig(filename)
        else:
            plt.tight_layout()
            plt.show()
        plt.clf()
        plt.cla()

    def print_and_plot_portfolio_weights(self, weights: dict,
                                         strategy: str, plot_num: int) -> None:
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['figure.figsize'] = (12, 6)

        print("\n-------- Weights for %s --------" % strategy)
        symbols = list(weights.keys())
        for k, v in weights.items():
            print(f"Symbol: {k}, Weight: {v:.4f}")

        # Plot
        width = 0.1
        x = np.arange(len(weights))
        plt.bar(x + (width * (plot_num - 1)) + 0.05,
                list(weights.values()), label=strategy, width=width)
        plt.xticks(x, symbols, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Symbols", fontsize=14)
        plt.ylabel("Weight in Portfolio", fontsize=14)
        plt.title("Portfolio Weights for Different Strategies", fontsize=14)
