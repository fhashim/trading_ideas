import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from helper_functions.get_data import download_data

def generate_random_portfolios(mean_returns, cov_matrix, num_portfolios=30000):
    num_assets = len(mean_returns)
    results = np.zeros((num_assets, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
    return results

def optimizer(tickers, start_date, end_date, required_return=0.02):

    df = download_data(tickers, start_date, end_date)
    
    # Calculate daily returns
    # returns = df['Close'].pct_change()

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
                weights @ mean_returns >= required_return, 
                weights >= 0]

    # Define the problem and solve
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Return optimization results
    return result, weights.value, mean_returns, cov_matrix


# tickers = ["VGSH","PBDAX","QQQ","GLD","SWPPX", "VUSTX", "LQD"]
tickers = ["AAPL", "TSLA", "MSFT"]

start_date = '2023-06-06' 
end_date = '2023-06-13'

results, weights, mean_returns, cov_matrix = optimizer(tickers, start_date, end_date, required_return=0.02)

# Generate random portfolios
random_portfolios = generate_random_portfolios(mean_returns, cov_matrix)

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(random_portfolios[1,:], random_portfolios[0,:], c=(random_portfolios[0,:] / random_portfolios[1,:]), marker='o')
plt.scatter([np.sqrt(results)], [np.sum(weights * mean_returns)], color='r', marker='*', s=600)
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier and Optimal Portfolio')
plt.show()

# Validate results from random portfolio
import pandas as pd
random_portfolios_df = pd.DataFrame(random_portfolios.T, columns = ['Returns', 'Std', 'Sharpe'])
random_portfolios_df[(random_portfolios_df.Returns >= 0.02) & (random_portfolios_df.Std < 0.12)]['Std'].sort_values()


#######
tickers = ["AAPL", "TSLA", "MSFT"]

start_date = '2023-06-06' 
end_date = '2023-06-13'

df = download_data(tickers, start_date, end_date)
    
# Calculate daily returns
# returns = df['Close'].pct_change()

# Calculate daily returns
returns = np.log(df['Close'] / df['Close'].shift()).dropna()

# Calculate mean returns and covariance
mean_returns = returns.mean()
cov_matrix = returns.cov()

from scipy.optimize import minimize 

def riskFunction(w):
    return np.dot(w.T, np.dot(cov_matrix, w))

w0 = [1/3.0, 1/3.0, 1/3.0]
bounds = ((0,1), (0,1), (0,1))

def checkMinimumReturns(w):
    rMin = 0.02
    RHS = rMin - np.sum(mean_returns*w)
    return RHS
def checkSumtoOne(w):
    return np.sum(w) - 1

constraints = ({'type':'eq', 'fun':checkMinimumReturns}, {'type':'eq', 'fun':checkSumtoOne})
w_opt = minimize(riskFunction, w0, method='SLSQP', bounds=bounds, constraints=constraints)


# Generate random portfolios
random_portfolios = generate_random_portfolios(mean_returns, cov_matrix)

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(random_portfolios[1,:], random_portfolios[0,:], c=(random_portfolios[0,:] / random_portfolios[1,:]), marker='o')
plt.scatter([np.sqrt(w_opt.fun)], [np.sum(w_opt.x * mean_returns)], color='r', marker='*', s=600)
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier and Optimal Portfolio')
plt.show()


### Read data
import pandas as pd
import yfinance as yf

tickers = ['CL', 'NG', 'USDPKR=X']
start_date = '2022-01-01'
end_date = '2023-06-19'
data = yf.download(tickers, start=start_date, end=end_date)
close_price = data['Close']



import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import yfinance as yf


# Stocks to consider
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'SPY', 'QQQ', 'VEA', 'GLD', 'SI', 'CL']

# Fetch stock data
start_date = '2022-01-01'
end_date = '2023-06-19'
df = yf.download(tickers=["600000.SS"], start=start_date, end=end_date)

returns = np.log(df['Close'] / df['Close'].shift()).dropna()

# reshaping the DataFrame
ticker_info = returns.melt(var_name='ticker', value_name='value')

ticker_info_grouped = ticker_info.groupby('ticker')['value'].agg(['mean', 'std']).reset_index()

# Calculate correlation matrix
corr_matrix = returns.corr()

# Initialize list
correlated_stocks = []

# Populate list
visited = set() # to keep track of the stocks that have been already considered

for i in range(corr_matrix.shape[0]):
    if tickers[i] not in visited: # if stock hasn't been considered yet
        temp = [tickers[i]] # start a new correlation group
        visited.add(tickers[i]) # mark the stock as visited
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i,j] > 0.7:  # Only consider correlations above a certain threshold
                temp.append(tickers[j])
                visited.add(tickers[j]) # mark the stock as visited
        correlated_stocks.append(temp)

# Print results
for sublist in correlated_stocks:
    print(f'Stocks that are highly correlated with each other: {sublist}')


corr_df = pd.DataFrame()
for corr_stocks, group in zip(correlated_stocks, range(0, len(correlated_stocks))):
    corr_df_temp = pd.DataFrame()
    corr_df_temp['Tickers'] = corr_stocks
    corr_df_temp['Group'] = group
    corr_df = pd.concat([corr_df_temp, corr_df])

corr_df = corr_df.reset_index(drop=True)

corr_df = corr_df.set_index('Tickers').join(ticker_info_grouped.set_index('ticker'))

risk_free_rate = 0
corr_df['Sharpe_Ratio'] = (corr_df['mean']  - risk_free_rate)/ corr_df['std']
corr_df_filtered = corr_df[corr_df.Sharpe_Ratio > 0]
corr_df_filtered = corr_df_filtered.sort_values(by='Sharpe_Ratio', ascending=False)
corr_df_filtered = corr_df_filtered.drop_duplicates(keep='first')


#### Final Testing Script ##########3

import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from helper_functions.get_data import download_data
from markowitz_portfolio_optimization.get_tickers import generate_tickers_list

def generate_random_portfolios(mean_returns, cov_matrix, risk_free_rate = 0.0, num_portfolios=10000):
    num_assets = len(mean_returns)
    results = np.zeros((num_assets, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results

def returns_optimizer(tickers, start_date, end_date, required_return=0.02):
    # Get data
    df = download_data(tickers, start_date, end_date)
    
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

def sharpe_optimizer(returns, risk_free_rate=0.0):
    
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
    lower_bound = 0.5 / len(tickers)
    t = (lower_bound,1)
    bounds = tuple(t for _ in range(len(tickers)))
    
    weights = np.array(len(tickers)*[1./len(tickers)])

    # Call minimizer
    opt_results = minimize(negate_sharpe_ratio, weights, constraints=cons, bounds=bounds, method='SLSQP')

    optimal_weights = opt_results.x
    ticker_weights = {}
    # optimal_weights
    for i in range(len(tickers)):
        ticker_weight = np.round(optimal_weights[i] * 100, 2)
        st = tickers[i]
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


# tickers = ["VGSH","PBDAX","QQQ","GLD","SWPPX", "VUSTX", "LQD"]
# tickers = ["AAPL", "TSLA", "MSFT"]
# tickers = ["SPY", "QQQ", "VEA"]
tickers, ticker_returns = generate_tickers_list(start_date = '2022-06-16', end_date = '2023-06-16', \
                                                user_tickers = [], n=5, correlation_value = 0.9, risk_free_rate = 0)
len(list(set(tickers)))
# Get data
# df = download_data(tickers, start_date = '2023-06-06', end_date = '2023-06-13')
ticker_weights, optimal_values, optimal_weights, mean_returns, cov_matrix, random_portfolios = \
    sharpe_optimizer(ticker_returns, risk_free_rate=0.0)

random_portfolio_df = pd.DataFrame()
random_portfolio_df['Returns'] = random_portfolios[1,:]
random_portfolio_df['Risk'] = random_portfolios[0,:]
random_portfolio_df['Sharpe'] = random_portfolios[2,:]
random_portfolio_df[random_portfolio_df.Sharpe >= optimal_values.get('Sharpe Ratio')].shape

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(random_portfolios[1,:], random_portfolios[0,:], c=random_portfolios[2,:] , marker='o', vmin=random_portfolios[2,:].min(), 
            vmax=max(random_portfolios[2,:].max(), optimal_values.get('Sharpe Ratio')))
plt.scatter([optimal_values['Volatility']], [optimal_values['Returns']], color='r', marker='*', s=600, vmin=random_portfolios[2,:].min(), 
            vmax=max(random_portfolios[2,:].max(), optimal_values.get('Sharpe Ratio')))
# plt.scatter(random_portfolios[1,:], random_portfolios[0,:], c=random_portfolios[2,:] , marker='o')
# plt.scatter([optimal_values['Volatility']], [optimal_values['Returns']], color='r', marker='*')
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier and Optimal Portfolio')
plt.show()

tickers_df = pd.DataFrame([ticker_weights]).T.rename(columns = {0:'Weightage'}).sort_values(by='Weightage', ascending=False)
tickers_df.sum()

# portfolio_return = np.sum(mean_returns * optimal_weights)
# portfolio_std_dev = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
# portfolio_return/portfolio_std_dev

#####
import requests
import bs4
import pandas as pd
import json
import yfinance as yf

url = "https://financialmodelingprep.com/api/v3/available-traded/list?apikey=1d323663f61f15ab293d38356a134f89"

response = requests.get(url)
soup = bs4.BeautifulSoup(response.text, 'html.parser')
universe = pd.DataFrame(json.loads(str(soup)))
universe.to_csv(r'data\universe.csv', index=False)

universe.exchangeShortName.unique()

universe.groupby(['exchangeShortName'])['exchange'].max().to_csv(r'data\exchange_names.csv')

exchanges = ['AMEX', 'AMS','ASE','ASX','BER','BRU','CPH',
             'EURONEXT','FGI','HKSE','LSE','MIL','MUN','NASDAQ',
             'SES','SHH','SHZ','SIX','STO','TSX','TWO']

exchanges = ['NASDAQ']
tickers_list = universe[(universe['exchangeShortName'].isin(exchanges)) & (universe['type'] == 'stock')].symbol.tolist()[:100]

start_date = '2023-06-15'
end_date = '2023-06-21'
df = yf.download(tickers=tickers_list, start=start_date, end=end_date)
df['Close'].shape