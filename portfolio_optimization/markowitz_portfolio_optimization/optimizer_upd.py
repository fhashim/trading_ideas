import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from IPython.display import display

from typing import List, Tuple
# from functools import cache  # Python 3.9 required

import yfinance as yf

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 9

TREASURY_BILL_RATE = 0.11  #%, Jan 2021
TRADING_DAYS_PER_YEAR = 250

# Needed for type hinting
class Asset:
  pass


def get_log_period_returns(price_history: pd.DataFrame):
  close = price_history['Close'].values  
  return np.log(close[1:] / close[:-1]).reshape(-1, 1)


# daily_price_history has to at least have a column, called 'Close'
class Asset:
  def __init__(self, name: str, daily_price_history: pd.DataFrame):
    self.name = name
    self.daily_returns = get_log_period_returns(daily_price_history)
    self.expected_daily_return = np.mean(self.daily_returns)
  
  @property
  def expected_return(self):
    return TRADING_DAYS_PER_YEAR * self.expected_daily_return

  def __repr__(self):
    return f'<Asset name={self.name}, expected return={self.expected_return}>'

  @staticmethod
  def covariance_matrix(assets: Tuple[Asset]):  # tuple for hashing in the cache
    product_expectation = np.zeros((len(assets), len(assets)))
    for i in range(len(assets)):
      for j in range(len(assets)):
        if i == j:
          product_expectation[i][j] = np.mean(assets[i].daily_returns * assets[j].daily_returns)
        else:
          product_expectation[i][j] = np.mean(assets[i].daily_returns @ assets[j].daily_returns.T)
    
    product_expectation *= (TRADING_DAYS_PER_YEAR - 1) ** 2

    expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
    product_of_expectations = expected_returns @ expected_returns.T

    return product_expectation - product_of_expectations


def random_weights(weight_count):
    weights = np.random.random((weight_count, 1))
    weights /= np.sum(weights)
    return weights.reshape(-1, 1)


class Portfolio:
    def __init__(self, assets: Tuple[Asset]):
        self.assets = assets
        self.asset_expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
        self.covariance_matrix = Asset.covariance_matrix(assets)
        self.weights = random_weights(len(assets))
    
    def unsafe_optimize_with_risk_tolerance(self, risk_tolerance: float):
        res = minimize(
        lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
        random_weights(self.weights.size),
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        ],
        bounds=[(0., 1.) for i in range(self.weights.size)]
        )   

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)
  
    def optimize_with_risk_tolerance(self, risk_tolerance: float):
        assert risk_tolerance >= 0.
        return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)
  
    def optimize_with_expected_return(self, expected_portfolio_return: float):
        res = minimize(
        lambda w: self._variance(w),
        random_weights(self.weights.size),
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return},
        ],
        bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self):
        # Maximize Sharpe ratio = minimize minus Sharpe ratio
        res = minimize(
        lambda w: -(self._expected_return(w) - TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
        random_weights(self.weights.size),
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        ],
        bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def _expected_return(self, w):
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]
    
    def _variance(self, w):
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

    @property
    def expected_return(self):
        return self._expected_return(self.weights)
    
    @property
    def variance(self):
        return self._variance(self.weights)

    def __repr__(self):
        return f'<Portfolio assets={[asset.name for asset in self.assets]}, expected return={self.expected_return}, variance={self.variance}>'
    
def yf_retrieve_data(tickers: List[str]):
    dataframes = []

    for ticker_name in tickers:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period='10y')

        if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
            history = history.iloc[1:]
    
        assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
        dataframes.append(history)
    
    return dataframes

# stocks = ['AAPL', 'AMZN', 'GOOG', 'BRK-B', 'JNJ', 'JPM']
stocks = ["AAPL", "TSLA", "MSFT"]
daily_dataframes = yf_retrieve_data(stocks)
assets = tuple([Asset(name, daily_df) for name, daily_df in zip(stocks, daily_dataframes)])

X = []
y = []

# Drawing random portfolios
for i in range(1000):
  portfolio = Portfolio(assets)
  X.append(np.sqrt(portfolio.variance))
  y.append(portfolio.expected_return)

plt.scatter(X, y, label='Random portfolios')

# Drawing the efficient frontier
X = []
y = []
for rt in np.linspace(-300, 200, 1000):
  portfolio.unsafe_optimize_with_risk_tolerance(rt)
  X.append(np.sqrt(portfolio.variance))
  y.append(portfolio.expected_return)

plt.plot(X, y, 'k', linewidth=3, label='Efficient frontier')

# portfolio.unsafe_optimize_with_risk_tolerance(0.05)
portfolio = Portfolio(assets)
portfolio.optimize_with_expected_return(0.25)
plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'g+', markeredgewidth=5, markersize=20, label='optimize_with_expected_return(0.25)')
plt.show()

portfolio.covariance_matrix
portfolio.asset_expected_returns
Asset(name, daily_df).daily_returns
##################


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt

TREASURY_BILL_RATE = 0.11  #%, Jan 2021
TRADING_DAYS_PER_YEAR = 250

def get_log_period_returns(price_history):
    close = price_history['Close'].values  
    return np.log(close[1:] / close[:-1]).reshape(-1, 1)

def calculate_expected_returns(daily_returns):
    return TRADING_DAYS_PER_YEAR * np.mean(daily_returns)

def covariance_matrix(assets_daily_returns, assets_expected_returns):
    daily_returns_concatenated = np.concatenate(assets_daily_returns, axis=1)
    product_expectation = np.cov(daily_returns_concatenated, rowvar=False) * (TRADING_DAYS_PER_YEAR - 1)
    product_of_expectations = np.outer(assets_expected_returns, assets_expected_returns)
    return product_expectation - product_of_expectations

def covariance_matrix(assets_daily_returns):
    # Transpose for correct computation of covariance matrix
    assets_daily_returns = np.array(assets_daily_returns).T
    return np.cov(assets_daily_returns) * (TRADING_DAYS_PER_YEAR - 1)





def random_weights(weight_count):
    weights = np.random.random(weight_count)
    weights /= np.sum(weights)
    return weights

def portfolio_variance(weights, covariance_matrix):
    return weights.T @ covariance_matrix @ weights

def portfolio_expected_return(weights, assets_expected_returns):
    return weights @ assets_expected_returns

def optimize_with_risk_tolerance(assets_expected_returns, covariance_matrix, risk_tolerance):
    num_assets = len(assets_expected_returns)

    res = minimize(
        lambda w: portfolio_variance(w, covariance_matrix) - risk_tolerance * portfolio_expected_return(w, assets_expected_returns),
        random_weights(num_assets),
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        ],
        bounds=[(0., 1.) for i in range(num_assets)]
    )

    assert res.success, f'Optimization failed: {res.message}'
    return res.x

def optimize_with_expected_return(assets_expected_returns, covariance_matrix, target_return):
    num_assets = len(assets_expected_returns)
    
    initial_weights = random_weights(num_assets)

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        {'type': 'eq', 'fun': lambda w: portfolio_expected_return(w, assets_expected_returns) - target_return},
    )

    bounds = [(0., 1.) for _ in range(num_assets)]
    
    res = minimize(
        lambda w: portfolio_variance(w, covariance_matrix),
        initial_weights,
        constraints=constraints,
        bounds=bounds
    )
    
    assert res.success, f'Optimization failed: {res.message}'
    return res.x

def yf_retrieve_data(tickers):
    dataframes = []

    for ticker_name in tickers:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period='10y')
        assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
        dataframes.append(history)
    
    return dataframes

stocks = ["AAPL", "TSLA", "MSFT"]
daily_dataframes = yf_retrieve_data(stocks)
assets_daily_returns = [get_log_period_returns(df) for df in daily_dataframes]
assets_expected_returns = [calculate_expected_returns(dr) for dr in assets_daily_returns]
cov_matrix = covariance_matrix(assets_daily_returns, assets_expected_returns)
cov_matrix = covariance_matrix(assets_daily_returns)
# Random portfolios for scatter plot
X = []
y = []
for _ in range(1000):
    weights = random_weights(len(stocks))
    X.append(np.sqrt(portfolio_variance(weights, cov_matrix)))
    y.append(portfolio_expected_return(weights, assets_expected_returns))

plt.scatter(X, y, label='Random portfolios')

# Efficient frontier
X = []
y = []
for rt in np.linspace(-300, 200, 1000):
    weights = optimize_with_risk_tolerance(assets_expected_returns, cov_matrix, risk_tolerance=rt)
    X.append(np.sqrt(portfolio_variance(weights, cov_matrix)))
    y.append(portfolio_expected_return(weights, assets_expected_returns))
weights = optimize_with_expected_return(assets_expected_returns, cov_matrix, 0.25)
portfolio_variance(weights, cov_matrix)

plt.plot(X, y, 'k', linewidth=3, label='Efficient frontier')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.legend()
plt.show()


for dr in assets_daily_returns:
   print(np.mean(dr))