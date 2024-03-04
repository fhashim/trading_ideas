import json
import yfinance as yf
import pandas as pd
from get_data.cnxn import server_access

def get_yfinance_data(tickers, start_date, end_date):
    with open('config/creds.json') as f:
        config = json.load(f)

    engine = server_access()
    sql = config["get_tickers_sql"]
    start_date = '2015-01-01'
    end_date = '2024-03-01'
    tickers_df = pd.read_sql(sql, engine)
    tickers_list = tickers_df.Ticker.tolist()
    df = yf.download(tickers_list, start=start_date, end=end_date)

    prices_df = pd.DataFrame()
    not_done_ticker = []
    for ticker in tickers_list:
        try:
            filtered_df = df.loc[:, (slice(None), ticker)]
            filtered_df = filtered_df.droplevel('Ticker', axis=1)
            filtered_df['Ticker'] = ticker
            # filtered_df.index.name = None
            filtered_df = filtered_df.rename_axis(None, axis=1)
            filtered_df= filtered_df.reset_index()
            filtered_df = filtered_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            prices_df = pd.concat([filtered_df, prices_df])
        except:
            not_done_ticker.append(ticker)

    prices_df = prices_df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    # Create a boolean mask for rows where column 'B' or 'C' is NaN
    mask = prices_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].isna().any(axis=1)

    # Filter the DataFrame using the mask
    prices_df = prices_df[~mask]
    prices_df[(prices_df.Ticker == 'ZURVY') & (prices_df.Date == '2015-03-02')]
    final_df = prices_df.drop_duplicates(subset=['Ticker', 'Date'])
    final_df['Open'] = final_df['Open'].round(4)
    final_df['High'] = final_df['High'].round(4)
    final_df['Low'] = final_df['Low'].round(4)
    final_df['Close'] = final_df['Close'].round(4)
    final_df['Adj Close'] = final_df['Adj Close'].round(4)
    final_df['Volume'] = final_df['Volume'].round(4)

    final_df[(final_df.Date >= '2015-01-01') & (final_df.Date <= '2015-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2016-01-01') & (final_df.Date <= '2016-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2017-01-01') & (final_df.Date <= '2017-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2018-01-01') & (final_df.Date <= '2018-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2019-01-01') & (final_df.Date <= '2019-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2020-01-01') & (final_df.Date <= '2020-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2021-01-01') & (final_df.Date <= '2021-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2022-01-01') & (final_df.Date <= '2022-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2023-01-01') & (final_df.Date <= '2023-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    
    final_df[(final_df.Date >= '2024-01-01') & (final_df.Date <= '2024-12-31')]\
        .to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)

    # final_df.to_sql('tbl_historical_prices_us', engine, if_exists='append', index=False)
    return df