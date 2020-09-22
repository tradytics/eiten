# Basic libraries
import os
import collections
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class DataEngine:
    def __init__(self, args):
        print("\n--> Data engine has been initialized...")
        self.args = args

        # Stocks list
        self.directory_path = str(os.path.dirname(os.path.abspath(__file__)))
        str_path = f"{self.directory_path}/{self.args.stocks_file_path}"
        self.stocks_file_path = str_path
        self.stocks_list = []

        # Load stock names in a list
        self.load_stocks_from_file()

        # Dictionary to store data. This will only store and save data if
        # the argument is_save_dictionary is 1.
        self.data_dictionary = {}

        # Data length
        self.stock_data_length = 0

    def load_stocks_from_file(self):
        """
        Load stock names from the file
        """
        print("Loading all stocks from file...")
        stocks_list = []
        with open(self.stocks_file_path, "r") as f:
            stocks_list = [str(item).strip() for item in f]

        # Load symbols
        stocks_list = list(sorted(set(stocks_list)))
        print("Total number of stocks: %d" % len(stocks_list))
        self.stocks_list = stocks_list

    def get_most_frequent_count(self, input_list):
        counter = collections.Counter(input_list)
        return list(counter.keys())[0]

    def _split_data(self, data):
        if self.args.is_test:

            return (data.iloc[:-self.args.future_bars]["Adj Close"].values,
                    data.iloc[-self.args.future_bars:]["Adj Close"].values)
        return data["Adj Close"].values, None

    def _format_symbol(self, s):
        x = s.upper()
        x = x.replace(".VN", ".V")
        if len(x.split(".")) > 2:
            x = x.replace(".", "-", 1)
        return x

    def get_data(self, symbol_raw):
        """
        Get stock data from yahoo finance.
        """
        symbol = self._format_symbol(symbol_raw)
        future_prices = None
        historical_prices = None
        # Find period
        if self.args.data_granularity_minutes == 1:
            period = "7d"
            interval = str(self.args.data_granularity_minutes) + "m"
        if self.args.data_granularity_minutes == 3600:
            period = "5y"
            interval = "1d"
        else:
            period = "30d"
            interval = str(self.args.data_granularity_minutes) + "m"

        # Get stock price
        try:
            # Stock price
            stock_prices = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False)
            # stock_prices = stock_prices.reset_index()

            if self.stock_data_length == 0:
                self.stock_data_length = stock_prices.shape[0]
            elif stock_prices.shape[0] != self.stock_data_length:
                raise Exception(f"{symbol}: Invalid Stock Length")

            if self.args.history_to_use == "all":
                # For some reason, yfinance gives some 0
                # values in the first index
                stock_prices = stock_prices.iloc[1:]
            else:
                stock_prices = stock_prices.iloc[-self.args.history_to_use:]

            historical_prices, future_prices = self._split_data(stock_prices)

        except Exception as e:
            print("Exception", e)
            return None, None

        return historical_prices, future_prices

    def collect_data_for_all_tickers(self):
        """
        Iterates over all symbols and collects their data
        """

        print("Loading data for all stocks...")
        data_dict = {"historical": pd.DataFrame(),
                     "future": pd.DataFrame()
                     }

        # Any stock with very low volatility is ignored.
        # You can change this line to address that.
        for i in tqdm(range(len(self.stocks_list))):
            symbol = self.stocks_list[i]
            try:
                historical_data, future_data = self.get_data(symbol)
                if historical_data is not None:
                    data_dict["historical"][symbol] = historical_data
                if future_data is not None:
                    data_dict["future"][symbol] = future_data
            except Exception as e:
                print("Exception", e)
                continue
        data_dict["historical"] = data_dict["historical"].fillna(1)
        data_dict["future"] = data_dict["future"].fillna(1)

        try:
            data_dict["historical"].to_csv("historical.csv")
            data_dict["future"].to_csv("future.csv")

        except Exception as e:
            print("Exception: ", e)

        # try:
        #     plt.style.use('seaborn-white')
        #     plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        #     plt.rcParams['axes.edgecolor'] = "#04383F"
        #     plt.rcParams['figure.figsize'] = (16, 9)
        #     data_dict["historical"].plot()
        #     plt.savefig("./output/gt_historical.png")
        #     data_dict["future"].plot()
        #     plt.savefig("./output/gt_future.png")
        #     plt.clf()
        # except Exception as e:
        #     print("Exception: ", e)

        return data_dict
