import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import yfinance as yf


import sys
sys.path.append("markowitz_portfolio_optimization")
from random_portfolio_generator import generate_random_portfolios


def returns_optimizer(tickers, start_date, end_date, required_return=0.02):

    # Get data
    df = yf.download(tickers, start=start_date, end=end_date)
    
    # Calculate daily returns
    returns = np.log(df['Close'] / df['Close'].shift()).dropna()

    # Calculate mean returns and covariance
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define the weights variables for cvxpy
    weights = cp.Variable(len(tickers))

    # Define the objective function
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

    # Define the constraints
    constraints = [cp.sum(weights) == 1, 
                weights @ mean_returns == required_return, 
                weights >= 0]

    # Define the problem and solve
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Return optimization results
    return result, weights.value, mean_returns, cov_matrix

def sharpe_optimizer(tickers, returns, risk_free_rate=0.0):
    """
    It is an optimization function that generates an 
    optimal allocation for the returns of all the 
    assets passed to this function. The function 
    ensures that 50% of the allocation is distributed 
    equally among all the assets, while the remaining 
    50% is distributed among the assets that would 
    result in the maximum Sharpe ratio. 

    Parameters:
        tickers (list): List of tickers passed for optimization
        returns (numpy.ndarray): Returns matrix for the list of ticker.
        risk_free_rate (float or int): Value for risk free rate which will be used to compute Sharpe ratio.

    Returns:
        ticker_weights (dict): Allocation weights for each selected asset (sums up to 1).
        optimal_values (dict): Expected optimal values for Return, Volatility, and Sharpe Ratio.
        optimal_weights (dict): Optimal weight allocation for all the selected assets.
        mean_returns (pandas.Series): Mean returns for all the selected assets.
        cov_matrix (pandas.DataFrame): Covariance matrix for all the selected assets.
        random_portfolios (numpy.ndarray): Array containing randomly generated portfolios 
                                            using mean returns and variance from selected assets.

    """

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
    lower_bound = 0.5 / len(tickers)
    t = (lower_bound,1)
    bounds = tuple(t for _ in range( len(tickers)))
    
    weights = np.array( len(tickers)*[1./ len(tickers)])

    # Call minimizer
    opt_results = minimize(negate_sharpe_ratio, weights, constraints=cons, bounds=bounds, method='SLSQP')

    optimal_weights = opt_results.x
    ticker_weights = {}
    # optimal_weights
    for i in range(returns.shape[1]):
        ticker_weight = np.round(optimal_weights[i] * 100, 2)
        st = tickers[i]
        ticker_weights[st] = np.round(float(ticker_weight),2)
    

    optimal_array = get_returns_volatility_sharpe(optimal_weights)
    keys = ['Returns', 'Volatility', 'Sharpe Ratio']
    values = list(optimal_array)

    optimal_values = dict(zip(keys, values))

    # Generate random portfolios
    random_portfolios = generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate=risk_free_rate)

    # Return optimization results
    return  ticker_weights, optimal_values, optimal_weights, mean_returns, cov_matrix, random_portfolios




