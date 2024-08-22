import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate = 0.0, num_portfolios=10000):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results

def mean_variance_optimizer(returns, risk_free_rate=0.0):
    
    # Calculate daily returns
    # returns = np.log(df['Close'] / df['Close'].shift()).dropna()

    # Calculate mean returns and covariance
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    def get_returns_volatility_sharpe(weights):
        weights = np.array(weights)
        returns = mean_returns.dot(weights)
        volatility = np.sqrt(weights.T.dot(cov_matrix.dot(weights)))
        sharpe_ratio = (returns - risk_free_rate) / volatility
        return returns, volatility, sharpe_ratio

    def negate_sharpe_ratio(weights):
        return get_returns_volatility_sharpe(weights)[-1] * -1
    
    # check sum of weights 
    def check_sum(weights):
        return np.sum(weights) - 1

    # Constraints for the optimization problem
    cons = {'type':'eq','fun':check_sum}

    # bounds on weights
    lower_bound = 0.5 / len(returns)
    t = (lower_bound,1)
    bounds = tuple(t for _ in range(len(returns)))
    
    weights = np.array(len(returns)*[1./len(returns)])

    # Call minimizer
    opt_results = minimize(negate_sharpe_ratio, weights, constraints=cons, bounds=bounds, method='SLSQP')

    optimal_weights = opt_results.x
    ticker_weights = {}
    # optimal_weights
    for i in range(len(returns)):
        ticker_weight = np.round(optimal_weights[i] * 100, 2)
        st = returns[i]
        ticker_weights[st] = np.round(float(ticker_weight),2)
    
    # ticker_weights = json.dumps(ticker_weights)

    optimal_array = get_returns_volatility_sharpe(optimal_weights)
    keys = ['Returns', 'Volatility', 'Sharpe Ratio']
    values = list(optimal_array)

    optimal_values = dict(zip(keys, values))
    # optimal_values = json.dumps(optimal_values)

    # Generate random portfolios
    random_portfolios = generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate=risk_free_rate)

    # Return optimization results
    return  ticker_weights, optimal_values, optimal_weights, mean_returns, cov_matrix, random_portfolios
