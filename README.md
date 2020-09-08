<p align="center">
  <img width="325" src="figures/normal-512x512.png">
</p>

# Eiten - Algorithmic Investing Strategies for Everyone
Eiten is an open source toolkit by [Tradytics](https://www.tradytics.com/) that implements various statistical and algorithmic investing strategies such as Eigen Portfolios, Minimum Variance Portfolios, Maximum Sharpe Portfolios, and Genetic Algorithms based Portfolios.

### Files Description
| Path | Description
| :--- | :----------
| eiten | Main folder.
| &boxur;&nbsp; figures | Folder to save data dictionaries for later use. 
| &boxur;&nbsp; stocks | Figures for this github repositories.
| &boxur;&nbsp; strategies | List of all the stocks that you want to analyze.
| data_loader.py | Module for loading data from yahoo finance.
| detection_engine.py | Main module for running anomaly detection on data and finding stocks with most unusual price and volume patterns.
| feature_generator.py | Generates price and volume return features as well as plenty of technical indicators.