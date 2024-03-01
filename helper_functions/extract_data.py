from binance.client import Client
import pandas as pd


client = Client("7T6Dk95WBZ3mE1Y4tH9bs4W0Nrx8tx7APQ3QnmObEuflk2jX9hZxr3jfrPxmPySe", 
               "LIH8o32jOzux4tGHkRQQg99SWCQel8ZpdBs6Y4cka82sns3naYeCxfhu5IvfeLrU")

# klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 Dec, 2017", "2 Apr, 2023")

# cols = ['open time',
# 'Open price',
# 'High price',
# 'Low price',
# 'Close price',
# 'Volume',
# 'Kline Close time',
# 'Quote asset volume',
# 'Number of trades',
# 'Taker buy base asset volume',
# 'Taker buy quote asset volume',
# 'Unused field, ignore.']

# raw_df = pd.DataFrame(klines, columns=[cols])

# raw_df.head()


# import pandas as pd
# import config
# from binance.client import Client
from datetime import timedelta
# import sqlalchemy as sa

# API_KEY = config.API_KEY
# SECRET_KEY = config.SECRET_KEY
# client = Client(API_KEY, SECRET_KEY)

date_range = pd.date_range('2017-01-01', '2023-04-02')

df_comp = pd.DataFrame()

for start_date in date_range:
    try:
        end_date = start_date + timedelta(hours=23, minutes=59, seconds=59)
        ticker = "BTCUSDT"
        candlesticks = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1MINUTE, str(start_date),
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


print(df_comp.head())

df_comp.to_csv('btc_usdt.csv', index=False)



