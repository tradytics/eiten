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