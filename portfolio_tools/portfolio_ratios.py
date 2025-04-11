import numpy as np

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate):
    """
    Calculate the Sharpe Ratio for a portfolio.

    Parameters:
    portfolio_returns (array-like): An array of portfolio returns.
    risk_free_rate (float): The risk-free rate of return.

    Returns:
    float: The Sharpe ratio of the portfolio.
    """
    # Convert the list of returns to a numpy array
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate the average return of the portfolio
    mean_return = np.mean(portfolio_returns)
    
    # Calculate the standard deviation (volatility) of the portfolio returns
    volatility = np.std(portfolio_returns)
    
    # Calculate the Sharpe ratio
    sharpe_ratio = (mean_return - risk_free_rate) / volatility
    
    return sharpe_ratio

def calculate_sortino_ratio(portfolio_returns, risk_free_rate):
    """
    Calculate the Sortino Ratio for a portfolio.

    Parameters:
    portfolio_returns (array-like): An array of portfolio returns.
    risk_free_rate (float): The risk-free rate of return.

    Returns:
    float: The Sortino ratio of the portfolio.
    """
    # Convert the list of returns to a numpy array
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate the average return of the portfolio
    mean_return = np.mean(portfolio_returns)
    
    # Calculate the downside deviation (volatility of negative returns)
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    # Calculate the Sortino ratio
    if downside_deviation == 0:
        return np.inf  # If there's no downside deviation, the ratio is infinite
    else:
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio
