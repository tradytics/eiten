<p align="center">
  <img width="325" src="figures/normal-512x512.png">
</p>

# Eiten - Algorithmic Investing Strategies for Everyone
Eiten is an open source toolkit by [Tradytics](https://www.tradytics.com/) that implements various statistical and algorithmic investing strategies such as Eigen Portfolios, Minimum Variance Portfolios, Maximum Sharpe Portfolios, and Genetic Algorithms based Portfolios.

### Files Description
| Path | Description
| :--- | :----------
| eiten | Main folder.
| &boxur;&nbsp; figures | Figures for this github repositories.
| &boxur;&nbsp; stocks | Folder to keep your stock lists that you want to use to create your portfolios.
| &boxur;&nbsp; strategies | A bunch if strategies implemented in python.
| backtester.py | Backtesting module that both backtests and forward tests all portfolios.
| data_loader.py | Module for loading data from yahoo finance.
| portfolio_manager.py | Main file that takes in a bunch of arguments and generates several portfolios for you.
| simulator.py | Simulator that uses historical returns and monte carlo to simulate future prices for the portfolios.
| strategy_manager.py | Manages the strategies implemented in the 'strategies' folder.

## Usage
### Packages
You will need to install the following package to train and test the models.
- [Scikit-learn](https://scikit-learn.org/)
- [Numpy](https://numpy.org/)
- [Tqdm](https://github.com/tqdm/tqdm)
- [Yfinance](https://github.com/ranaroussi/yfinance)
- [Pandas](https://pandas.pydata.org/)
- [Scipy](https://www.scipy.org/install.html)

You can install all packages using the following command. Please note that the script was written using python3.

```
pip install -r requirements.txt
```

### Build your portfolios
Let us see how we can use all the strategies given in the toolkit to build our portfolios. The first thing you need to do is modify the stocks.txt file in the stocks folder and add the stocks of your choice. It is recommended to keep the list small i.e anywhere between 5 to 50 stocks should be fine.