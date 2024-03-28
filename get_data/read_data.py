import json
from datetime import timedelta
from binance.client import Client
import pandas as pd
from get_data.cnxn import server_access


def get_ticker_data_pak(ticker, start_date, end_date):
    engine = server_access()
    sql = '''   select Date, Ticker, Open, High, Low, close Close, Volume from stocks.tbl_historical_prices_pak 
                where ticker = '{}'
                and Date between '{}' and '{}'
                order by Date asc;     '''.format(ticker, start_date, end_date)
    
    df = pd.read_sql(sql, engine)

    return df

def get_tickers_list():
    engine = server_access()
    sql = '''   select distinct(Ticker) from stocks.tbl_historical_prices_pak; '''
    
    df = pd.read_sql(sql, engine)
    ticker_list = df.Ticker.tolist()

    return ticker_list

def get_ticker_data_us(ticker, start_date, end_date):
    engine = server_access()
    sql = '''   select Date, Ticker, Open, High, Low, close Close, Volume from stocks.tbl_historical_prices_us 
                where ticker = '{}'
                and Date between '{}' and '{}'
                order by Date asc;     '''.format(ticker, start_date, end_date)
    
    df = pd.read_sql(sql, engine)

    return df

def get_ticker_data_binance(ticker, range_start_date, range_end_date):

    # load config file to extract necessary information
    with open('config/creds.json') as f:
        configs = json.load(f)
        f.close()

    read_api = configs['binance_read_api_key']
    read_api_secret = configs['binance_read_api_secret']

    client = Client(read_api, read_api_secret)
    date_range = pd.date_range(range_start_date, range_end_date)

    df_comp = pd.DataFrame()
    ticker = "{}USDT".format(ticker)

    for start_date in date_range:
        try:
            end_date = start_date + timedelta(hours=23, minutes=59, seconds=59)
            candlesticks = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_15MINUTE, str(start_date),
                                                        str(end_date))

            for candlestick in candlesticks:
                candlestick[0] = candlestick[0] / 1000

            for candlestick in candlesticks:
                candlestick[6] = candlestick[6] / 1000

            df = pd.DataFrame(candlesticks, columns=['OpenEpoch', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseEpoch',
                                                    'QuoteAssetVolume', 'NumberOfTrades', 'TBBAV', 'TBQAV', 'ign'])

            df['OpenDate'] = pd.to_datetime(df['OpenEpoch'], unit='s')
            df['CloseDate'] = pd.to_datetime(df['CloseEpoch'], unit='s')
            df['ticker'] = ticker

            df = df[['OpenDate', 'CloseDate', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteAssetVolume',
                    'NumberOfTrades', 'TBBAV', 'TBQAV', 'OpenEpoch', 'CloseEpoch']]
            
            df_comp = pd.concat([df, df_comp])

        except:
            print(start_date)

    return df_comp