import pandas_datareader.data as web
import pandas as pd
import matplotlib
import yfinance as yf

import scipy

from scipy.optimize import curve_fit
import numpy as np
from numpy import array
import math

import seaborn as sns
from bs4 import BeautifulSoup
import requests
import csv
import selenium

# generate random Gaussian values
from random import seed
from random import gauss

import matplotlib.pyplot as plt


#Below link was initial source of code. 

#https://towardsdatascience.com/how-to-construct-an-efficient-portfolio-using-the-modern-portfolio-theory-in-python-5c5ba2b0cff4

#https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
#Below are two part series
#https://medium.com/python-data/effient-frontier-in-python-34b0c3043314
#https://medium.com/python-data/efficient-frontier-portfolio-optimization-with-python-part-2-2-2fe23413ad94

sym = yf.Ticker("MSFT")

# get stock info
sym.info

# get historical market data
hist = sym.history(period="max")

port_tickers=["VGSH","PBDAX","QQQ","GLD","SWPPX", "VUSTX", "LQD"]

data = yf.download("EEM, QQQ,^NDX,GLD,GC=F,^GSPC,VHGEX,EFA, VUSTX, LQD", start="2000-01-01", end="2021-04-30")
data.to_csv("big_data.csv")
port_name:["vanguard short treasuries","Vanguard corp bond" , "S&P500", "Gold ETF", "Vanguard Inter treasuries"]
portfolio_comp=["cash", "corp bonds", "Equities","inflation", "Risk Free bonds"]

print(hist)
print(len(port_tickers))
hist_close=[]
price_arr=[]


for x in port_tickers:
    print("symbol loop",x)
    sym = yf.Ticker(x)
    hist = sym.history(period="10y")
    print("date", hist)

    hist_close.insert(1,hist["Close"].values)
    
    #hist_close_ser=pd.Series(hist_close)
    #priceDF=pd.DataFrame(hist_close, columns=[x])
print("loop",hist_close)
print("after loop",hist_close)

priceDF=pd.DataFrame(hist_close)
priceDF=priceDF.transpose()

priceDF.columns=port_tickers

    
    #np.insert(price_arr, x, hist_close) 


#print("price array",price_arr)

priceDF.to_excel("prices.xlsx")

#below covariance matrix isn't used
covar_matrix = [(.8, .9, .1,.7 ),

           (.8, .9, .1,.7),

           (.8, .9, .1,.7),
           (.8, .9, .1,.7)];


#dataFrame1 = pd.DataFrame(data=matrix1);

covar_DF=pd.DataFrame(data=covar_matrix);
#print(covar_DF)




model_port_tick=["SPY","QQQ", "EFA", "VWO", "VNQ"];
model_port_FI_tick=["VSGH","VCIT", "VUSTX", "BNDX"]
model_port_co_tick=["GLD", "SCHP"]




Exp_Returns=[]



#port_prices = web.DataReader(port_tickers,data_source="yahoo", start='01/03/2000')['Adj Close']

port_prices=pd.read_csv("port_prices.csv")

print("prices baby",port_prices)
big_portfolio=pd.read_csv("big_data_input.csv", index_col='date', parse_dates=True)

big_portfolio.interpolate(method ='linear', limit_direction ='forward')
print("portfolio input detail")
print(big_portfolio.describe())

big_portolio_returns=big_portfolio.pct_change(periods=30)



print("big_portfolio returns correlation")
print(big_portolio_returns.corr())

sns.heatmap(big_portolio_returns.corr(), cmap='coolwarm', annot=True)

plt.show()
p=sns.heatmap(big_portolio_returns.corr(), cmap='coolwarm', annot=True)
p.get_figure().savefig('corr_heatmap.png')

#consider log returns and using shift function:


#log_ret=np.log((close_price/closeprice.shift(1)))

#below uses monthly returns, non annualized.
port_returns= port_prices.pct_change(periods=30)
port_returns_annual= port_prices.pct_change(periods=250)


port_returns=(port_returns+1)**12-1


print("annual returns")
print(port_returns_annual.describe())



#port_returns=(port_returns+1)**365-1


port_corr=port_returns.corr()
port_cov=port_returns.cov()



print(port_prices.head())

print(port_returns.head())
mean_ret=port_returns.mean(axis=0)
print("mean_ret", mean_ret)

print("corr matrix", port_corr)

print("cov Matrix", port_cov)

#portfolio_risk

#portfolio_returns



port_weight=[0.4,0.2,0.2,0.1,0.1]


Historical_returns=[]


#result = dataFrame1.dot(dataFrame2);



#Generating a random matrix of 1000 rows and x Columns (depending on number of investments)

matrix = np.random.rand(1000,5)

print("matrix", matrix)
print()

#Converting to a data frame
matrix_df = pd.DataFrame(matrix, columns = port_returns.columns)

matrix_df.to_excel("random_matrix.xlsx")

print("matrix df","\n", matrix_df.head(100))

matrix_sum = matrix_df.sum(axis = 1)
#Calculating portfolio weights
weights  = matrix_df.divide(matrix_sum , axis ="rows")
#transpose
#weights_t= np.transpose(weights)
weights_t= np.transpose(weights)
#weights_t_df=df = pd.Series(weights_t)

print("transposed weights", weights_t)

#Using the portfolio return formula using dot (replaced with matmul) which is a matrix multiplication function

portfolio_return = np.matmul(weights, mean_ret)

print(portfolio_return)


#Variance covariance
cov_mat = port_returns.cov()
print("cov mat", cov_mat)
portfolio_risk = []
#below loop runs the simulation
for one_port in range(weights.shape[0]):
    #print("weights shape", weights.shape[0])
    #print("one_port", one_port)

    #risk is calculated by multiplying the randomly generated weights by the weights again and then taking square root. 

    risk = np.sqrt(np.matmul(weights.iloc[one_port,:],np.matmul(cov_mat,weights_t.iloc[:,one_port])))
    portfolio_risk.append(risk)
    #print("risk", portfolio_risk)

#https://stackoverflow.com/questions/37234163/how-to-add-a-line-of-best-fit-to-scatter-plot

plt.figure(figsize = (10,8))
plt.scatter(portfolio_risk, portfolio_return)
plt.plot(np.polyfit(portfolio_risk, portfolio_return, 2))

plt.xlabel("Portfolio Risk - Standard Deviation")
plt.ylabel("Portfolio Return")
plt.savefig("efficient_frontier.png")
plt.show()


#below adds a fitted line using seaborn library. Can also me achieved using polyfit
#in matplotlib
#https://stackoverflow.com/questions/37234163/how-to-add-a-line-of-best-fit-to-scatter-plot


p=sns.regplot(portfolio_risk,portfolio_return)

p.set_xlabel("Risk", fontsize = 20)
p.set_ylabel("Returns", fontsize = 20)
plt.show()


#converting to a csv file
portfolio_risk = pd.DataFrame(portfolio_risk, columns = ["portfolio risk"])
portfolio_return = pd.DataFrame(portfolio_return, columns = ["portfolio return"])
random_portfolio = pd.concat([portfolio_return, portfolio_risk, weights], axis =1)
random_portfolio.to_csv("Random_Portfolios.csv")


print("portfolio optimum summary")
print(random_portfolio.describe())

#Below is from a separate blog post


def efficient_return (mean_returns, cov_matrix, target):
    num_assets = len (mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return (weights):
        return portfolio_annualised_performance (weights, mean_returns, cov_matrix) [1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return (x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum (x) - 1})
    bounds = tuple ((0,1) for asset in range (num_assets))
    result = sco.minimize (portfolio_volatility, num_assets * [1./num_assets,], args = args, method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result


def efficient_frontier (mean_returns, cov_matrix, returns_range):
    efficient = []
    for ret in returns_range:
        efficient.append (efficient_return (mean_returns, cov_matrix, ret))
    return efficiencies