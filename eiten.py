# !/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load our modules
from data_loader import DataEngine
from backtester import BackTester
from utils import random_matrix_theory_based_cov
from utils import dotdict, get_predicted_returns, get_exp_returns
from utils import get_price_deltas, get_log_returns
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
        if self.args.history_to_use != "all":
            self.args.history_to_use = int(self.args.history_to_use)

        # Data dictionary
        self.data_dict = {}  # {"market": args.market_index}
        self.market_data = {}

        print('\n')

    def load_data(self):
        """
        Loads data needed for analysis
        """
        # Gather data for all stocks in a dictionary format
        # Dictionary keys will be -> historical, future
        de = DataEngine(self.args)
        self.data_dict = de.collect_data_for_all_tickers()
        p, f = de.get_data(self.args.market_index)

        self.market_data["historical"] = pd.DataFrame(
            columns=[self.args.market_index], data=p)
        self.market_data["future"] = pd.DataFrame(
            columns=[self.args.market_index], data=f)
        # Get return matrices and vectors
        return self.data_dict

    def _test(self, direction):
        # Back test
        print("\n*& Backtesting the portfolios...")
        assert direction in ["historical", "future"], "Invalid direction!"

        return pd.DataFrame(columns=self.portfolios.columns,
                            data=BackTester.get_test(
                                self.portfolios,
                                self.data_dict,
                                direction,
                                self.args.only_long))

    def _monte_carlo(self, span):
        self.data_dict["sim"] = BackTester.simulate_future_prices(
            self.data_dict, get_predicted_returns, span)
        return pd.DataFrame(columns=self.portfolios.columns,
                            data=BackTester.get_test(
                                self.portfolios,
                                self.data_dict,
                                "sim",
                                self.args.only_long))
        # BackTester.plot_test(title="Simulated Future Returns",
        #                      xlabel="Bars (Time Sorted)",
        #                      ylabel="Cumulative Percentage Return",
        #                      df=df)

    def run_strategies(self):
        """
        Run strategies, back and future test them, and simulate the returns.
        """
        self.load_data()

        # Calculate covariance matrix
        log_returns = get_log_returns(self.data_dict["historical"])
        cov_matrix = log_returns.cov()

        # Use random matrix theory to filter out the noisy eigen values
        if self.args.apply_noise_filtering:
            print("\nFiltering noise from cov matrix\n")
            cov_matrix = random_matrix_theory_based_cov(log_returns)

        pred_returns = get_predicted_returns(self.data_dict["historical"])
        perc_returns = get_price_deltas(self.data_dict["historical"])

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
        self.portfolios = pd.DataFrame.from_dict(self.portfolios)

        # Print weights
        print("\n*% Printing portfolio weights...")
        p_count = 1
        print(self.portfolios)
        self.draw_plot("output/weights.png", (p_count, 6))
        self._test("historical")
        self.draw_plot("output/back_test.png")

        if self.args.is_test:
            self._test("future")
        self.draw_plot("output/future_tests.png")

        # Simulation
        print("\n+$ Simulating future prices using monte carlo...")
        self._monte_carlo(self.args.future_bars)
        self.draw_plot("output/monte_carlo.png")
        return

    def draw_plot(self, filename="output/graph.png", figsize=(12, 6)):
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
        plt.rcParams['figure.figsize'] = figsize

        plt.grid()
        plt.legend(fontsize=14)
        if self.args.save_plot:
            plt.savefig(filename)
        else:
            plt.tight_layout()
            plt.show()
        # plt.cla()
        plt.clf()

    def print_and_plot_portfolio_weights(self,
                                         weights: dict, strategy: str,
                                         plot_num: int, figsize=(12, 6)):

        print("\n-------- Weights for %s --------" % strategy)
        symbols = list(weights.keys())
        for k, v in weights.items():
            print(f"Symbol: {k}, Weight: {v:.4f}")

        # Plot
        width = 0.1
        x = np.arange(len(weights))
        plt.bar(x + (width * (plot_num - 1)) + 0.05,
                list(weights.values()), label=strategy, width=width)
        plt.xticks(x, symbols, rotation=90)
        plt.yticks(fontsize=14)
        plt.xlabel("Symbols", fontsize=14)
        plt.ylabel("Weight in Portfolio", fontsize=14)
        plt.title("Portfolio Weights for Different Strategies", fontsize=14)
