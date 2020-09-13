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
        self.stock_data_length = []

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

            return (data.iloc[:-self.args.future_bars],
                    data.iloc[-self.args.future_bars:])

    def get_data(self, symbol):
        """
        Get stock data from yahoo finance.
        """
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
            stock_prices = stock_prices.reset_index()
            try:
                stock_prices = stock_prices.drop(columns=["Adj Close"])
            except Exception as e:
                print("Exception", e)

            data_length = stock_prices.shape[0]
            self.stock_data_length.append(data_length)

            if self.args.history_to_use == "all":
                # For some reason, yfinance gives some 0
                # values in the first index
                stock_prices = stock_prices.iloc[1:]
            else:
                stock_prices = stock_prices.iloc[-self.args.history_to_use:]

            historical_prices, future_prices = self._split_data(stock_prices)
            print(f"Data separation\nH:{historical_prices.shape[0]}\nF:{future_prices.shape[0]}")

        except Exception as e:
            print("Exception", e)

        return historical_prices, future_prices

    def collect_data_for_all_tickers(self):
        """
        Iterates over all symbols and collects their data
        """

        print("Loading data for all stocks...")
        data_dict = {}

        # Any stock with very low volatility is ignored.
        # You can change this line to address that.
        for i in tqdm(range(len(self.stocks_list))):
            symbol = self.stocks_list[i]
            try:
                historical_data, future_data = self.get_data(symbol)
                if historical_data is not None:
                    data_dict[symbol] = {
                        "historical": historical_data,
                        "future": future_data
                    }
            except Exception as e:
                print("Exception", e)
                continue

        return self.clean_data(data_dict)

    def clean_data(self, data_dict):
        """
        Remove bad data i.e data that had some errors while scraping or 
        feature generation

        """

        length_dictionary = collections.Counter(
            [data_dict[i]["historical"].shape[0] for i in data_dict])
        std_len = length_dictionary.most_common(1)[0][0]

        return {i: data_dict[i] for i in data_dict if data_dict[i]["historical"].shape[0] == std_len}
