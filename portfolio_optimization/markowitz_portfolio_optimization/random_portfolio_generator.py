import numpy as np

def generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios=10000):
    """
    Generates random portfolios based on the mean returns, covariance matrix, and risk-free rate.

    Parameters:
        mean_returns (numpy.ndarray): Array of mean returns for each asset.
        cov_matrix (numpy.ndarray): Covariance matrix of the assets.
        risk_free_rate (float): Risk-free rate used in the portfolio calculations.
        num_portfolios (int, optional): Number of random portfolios to generate. Default is 10000.

    Returns:
        numpy.ndarray: Array of portfolio results with shape (3, num_portfolios), where:
            - Row 1 contains the portfolio returns.
            - Row 2 contains the portfolio standard deviations.
            - Row 3 contains the portfolio Sharpe ratios.

    """
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