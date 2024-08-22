import pandas as pd
import numpy as np
import yfinance as yf


def generate_tickers_list(start_date, end_date, user_tickers = [], n=20, correlation_value = 0.8, risk_free_rate = 0, 
                          only_user_tickers = False):

    """
    This function generates a list of assets and their corresponding returns 
    after removing assets that are highly correlated or have a negative Sharpe ratio. 
    The default value to filter out highly correlated stocks is 0.8. 
    This means that assets with a correlation value of 0.8 or greater 
    are considered highly correlated, and only one asset with the highest 
    Sharpe ratio is selected from this group. By default, 
    the only_user_ticker parameter is set to False, which means 
    that the function uses assets provided by the users along with 
    ETFs and S&P500 assets to filter out high-performing stocks. 
    The parameter n represents the number of tickers to be returned. 
    By default, 20 assets are returned, but this can be changed by the user. 
    It is important to note that when only_user_tickers is set to True, 
    the n parameter is not used, and all tickers are returned after 
    filtering based on the aforementioned criteria. 
    If the start date for any asset(s) does not have any data, 
    those rows are dropped, and only the rows with available data are kept. 
    For example, if the series for Stock A starts from 2015-01-01 and the 
    series for Stock B starts from 2016-01-01, all computations are 
    performed starting from 2016-01-01, even if the user specifies a start date of 2015-01-01.

    Parameters:
        start_date (str or datetime): Start date for collecting historical data.
        end_date (str or datetime): End date for collecting historical data.
        user_tickers (list, optional): List of user-defined tickers. Default is an empty list.
        n (int, optional): Number of tickers to be generated. Default is 20.
        correlation_value (float, optional): Minimum correlation value between generated tickers. Default is 0.8.
        risk_free_rate (float, optional): Value for the risk-free rate used in optimization. Default is 0.
        only_user_tickers (bool, optional): Flag indicating whether to include only user-defined tickers.
                                            If True, other generated tickers will be excluded. Default is False.

    Returns:
        tickers_list (list): List of generated tickers.
    
    Notes:
    - The function filters out assets that are highly correlated or have a negative Sharpe ratio.
    - Assets provided by the user, along with ETFs and S&P500 assets, are used for filtering.
    - The number of tickers returned can be customized using the `n` parameter.
    - If `only_user_tickers` is True, `n` parameter is ignored, and all tickers are returned after filtering.
    - The start date of each asset's series is considered, and computation begins from the latest common start date
        where data is available, even if the user specifies an earlier start date.
    """

    if only_user_tickers:
        tickers = list(set(user_tickers))
    else:
        etfs = pd.read_csv(r'data\etfs.csv')
        sp500 = pd.read_csv(r'data\sp500.csv')

        tickers_sp500 = sp500['Symbol'].tolist()
        tickers_etfs = etfs['Ticker'].tolist()

        tickers = tickers_sp500 + tickers_etfs + user_tickers
        tickers = list(set(tickers))

    df = yf.download(tickers, start=start_date, end=end_date)
    df_close = df['Close'].fillna(method='ffill')
    df_close = df['Close'].dropna(axis=1)
    returns = (np.log(df_close / df_close.shift()).dropna()) * 252

    tickers = returns.columns.tolist()

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
                if corr_matrix.iloc[i,j] > correlation_value:  # Only consider correlations above a certain threshold
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

    
    corr_df['Sharpe_Ratio'] = (corr_df['mean']  - risk_free_rate)/ corr_df['std']
    corr_df_filtered = corr_df[corr_df.Sharpe_Ratio > 0]
    corr_df_filtered = corr_df_filtered.sort_values(by='Sharpe_Ratio', ascending=False)
    # corr_df_filtered = corr_df_filtered.drop_duplicates(keep='first')

    corr_df_filtered = corr_df_filtered.reset_index().rename(columns = {'index':'Tickers'})
    # corr_df_filtered = corr_df_filtered.groupby('Group')[ 'Sharpe_Ratio'].max()
    corr_df_filtered = corr_df_filtered.drop_duplicates(keep='first', subset='Group')
    if only_user_tickers:
        tickers_list = corr_df_filtered.Tickers.tolist()
    else:
        corr_df_filtered = corr_df_filtered.iloc[:n,]
        tickers_list = list(set(corr_df_filtered.Tickers.tolist()))

    ticker_returns = returns[tickers_list]

    return tickers_list, ticker_returns, corr_matrix, corr_df