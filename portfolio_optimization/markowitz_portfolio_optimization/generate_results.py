import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("markowitz_portfolio_optimization")

from get_tickers import generate_tickers_list
from optimizer import sharpe_optimizer

# user_tickers = ["VGSH","PBDAX","QQQ","GLD","SWPPX", "VUSTX", "LQD", "AAPL", "TSLA", "MSFT"]
user_tickers = ["VGSH","PBDAX","QQQ","GLD","SWPPX", "VUSTX", "LQD"]

risk_free_rate = 0.03
tickers, ticker_returns, corr_matrix, corr_df = generate_tickers_list(start_date = '2003-06-16', end_date = '2023-06-16', \
                                                user_tickers = user_tickers, n=5, correlation_value = 0.9, 
                                                risk_free_rate = risk_free_rate, \
                                                only_user_tickers = True)

ticker_weights, optimal_values, optimal_weights, mean_returns, cov_matrix, random_portfolios = \
    sharpe_optimizer(tickers, ticker_returns, risk_free_rate=risk_free_rate)
print("--------------------------------------------")
print("Tickers passed by user: " + str(user_tickers))
print("--------------------------------------------")
print("Selected tickers: " + str(tickers))
print("--------------------------------------------")
print("Ticker Correlation")
print(corr_matrix)
print("--------------------------------------------")
print("Ticker returns")
print(ticker_returns)
print("--------------------------------------------")
tickers_df = pd.DataFrame([ticker_weights]).T.rename(columns = {0:'Weightage'}).sort_values(by='Weightage', ascending=False)
# tickers_df.sum()
print('Portofolio Distirbution')
print(tickers_df)
print("--------------------------------------------")
print("Portfolio Returns "+ str(np.round(optimal_values.get('Returns'),5)))
print("--------------------------------------------")
print("Portfolio Volatility "+ str(np.round(optimal_values.get('Volatility'),5)))
print("--------------------------------------------")
print("Portfolio Sharpe Ratio "+ str(np.round(optimal_values.get('Sharpe Ratio'),5)))
print("--------------------------------------------")
print('Sharpe Ratio for all tickers:')
print(corr_df.sort_values(by=['Sharpe_Ratio'], ascending=False)['Sharpe_Ratio'])
print("--------------------------------------------")


random_portfolio_df = pd.DataFrame()
random_portfolio_df['Returns'] = random_portfolios[1,:]
random_portfolio_df['Risk'] = random_portfolios[0,:]
random_portfolio_df['Sharpe'] = random_portfolios[2,:]
# random_portfolio_df[random_portfolio_df.Sharpe >= optimal_values.get('Sharpe Ratio')].shape

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(random_portfolios[1,:], random_portfolios[0,:], c=random_portfolios[2,:] , marker='o', vmin=random_portfolios[2,:].min(), 
            vmax=max(random_portfolios[2,:].max(), optimal_values.get('Sharpe Ratio')))
plt.scatter([optimal_values['Volatility']], [optimal_values['Returns']], color='r', marker='*', s=600, vmin=random_portfolios[2,:].min(), 
            vmax=max(random_portfolios[2,:].max(), optimal_values.get('Sharpe Ratio')))

plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier and Optimal Portfolio')
plt.show()

