import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load our modules
from data_loader import DataEngine
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
        return f(self.data_dict["historical"])

    def _get_perc_returns(self):
        return self._get_abstract_returns(self._get_price_delta)

    def _get_log_return(self, data: pd.DataFrame) -> np.ndarray:
        return np.log((data / data.shift())[1:])

    def _get_log_returns(self):
        return self._get_abstract_returns(self._get_log_return)

    def _get_predicted_return(self, data: pd.DataFrame) -> np.ndarray:
        return self._get_price_delta(data).div(
            np.array(np.arange(len(data) - 1, 0, -1)), axis=0).mean(axis=0)

    def _get_predicted_returns(self):
        return self._get_abstract_returns(self._get_predicted_return)

    def load_data(self):
        """
        Loads data needed for analysis
        """
        # Gather data for all stocks in a dictionary format
        # Dictionary keys will be -> historical, future
        de = DataEngine(self.args)
        self.data_dict = de.collect_data_for_all_tickers()
        p, f = de.get_data(self.args.market_index)

        print("Market", p.shape, f.shape)
        self.market_data["historical"] = pd.DataFrame(
            columns=[self.args.market_index], data=p)
        self.market_data["future"] = pd.DataFrame(
            columns=[self.args.market_index], data=f)
        # Get return matrices and vectors
        return self.data_dict

    def _backtest(self):
        # Back test
        print("\n*& Backtesting the portfolios...")

        df = pd.DataFrame(columns=list(self.portfolios.keys()))
        for i in self.portfolios:
            df[i] = BackTester.get_test(
                self.portfolios[i],
                self.data_dict,
                "historical",
                self.args.only_long)
        mp = BackTester.get_market_returns(self.market_data, "historical")
        BackTester.plot_test(title="Backtest Results",
                             xlabel="Bars (Time Sorted)",
                             ylabel="Cumulative Percentage Return",
                             df=df)
        BackTester.plot_market(mp)

    def _futuretest(self):
        print("\n#^ Future testing the portfolios...")
        # Future test
        df = pd.DataFrame(columns=list(self.portfolios.keys()))
        for i in self.portfolios:
            df[i] = BackTester.get_test(self.portfolios[i],
                                        self.data_dict,
                                        "future",
                                        self.args.only_long)
        mp = BackTester.get_market_returns(self.market_data, "future")
        BackTester.plot_test(title="Future Test Results",
                             xlabel="Bars (Time Sorted)",
                             ylabel="Cumulative Percentage Return",
                             df=df)
        BackTester.plot_market(mp)

    def _monte_carlo(self):
        df = pd.DataFrame(columns=list(self.portfolios.keys()))
        self.data_dict["sim"] = BackTester.simulate_future_prices(
            self.data_dict, 30)
        for i in self.portfolios:
            df[i] = BackTester.get_test(self.portfolios[i],
                                        self.data_dict,
                                        "sim",
                                        self.args.only_long)
            BackTester.plot_test(title="Simulated Future Returns",
                                 xlabel="Bars (Time Sorted)",
                                 ylabel="Cumulative Percentage Return",
                                 df=df)

    def run_strategies(self):
        """
        Run strategies, back and future test them, and simulate the returns.
        """
        self.load_data()
        print("historical", self.data_dict["historical"].shape)
        print("future", self.data_dict["future"].shape)

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
                perc_returns=perc_returns,
                long_only=self.args.only_long)
            self.portfolios[name] = weights

        # Print weights
        print("\n*% Printing portfolio weights...")
        p_count = 1
        for i in self.portfolios:
            self.print_and_plot_portfolio_weights(
                self.portfolios[i], i, plot_num=p_count)
            p_count += 1
        self.draw_plot("output/weights.png")
        self._backtest()

        if self.args.is_test:
            self._futuretest()
        self.draw_plot("output/future_tests.png")

        # Simulation
        print("\n+$ Simulating future prices using monte carlo...")
        self._monte_carlo()
        self.draw_plot("output/monte_carlo.png")
        return

    def draw_plot(self, filename="output/graph.png"):
        """
        Draw plots
        """
        # Styling for plots
        plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['axes.titlesize'] = "large"
        plt.rcParams['axes.labelsize'] = "medium"
        plt.rcParams['lines.linewidth'] = 2
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
