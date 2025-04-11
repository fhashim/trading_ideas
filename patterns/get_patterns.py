import pandas as pd
import numpy as np
from get_data.read_data import get_ticker_data_pak


def hammer(df):

    # df = get_ticker_data_pak(ticker, start_date, end_date)

    # df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
    df['Date'] = pd.to_datetime(df['Date'])

    float_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    for col in float_cols:
        df[col] = df[col].astype(float)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['x_val'] = (0.618 * (df['High'] - df['Low'])) + df['Low']
    df['signal'] = np.where((df['Open'] < df['Close']) & 
                            (df['Open'] > df['x_val']), 1, 0)
    # df['next_signal'] = np.where(df['signal'].shift() == 1, -1, 0)
    df['next_signal'] = np.where((df['signal'].shift() == 1) & (df['Open'] < df['Close']), 1, 0)
    df['signal_rev'] = np.where((df['next_signal'] == 1) & (df['signal'].shift() == 1), 1, 0)
    # df['signal'] = np.where(df.next_signal == -1, -1, df.signal)
    # df.set_index('Date', inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal_rev']]
    df = df.rename(columns={'signal_rev':'signal'})

    return df
    
def engulfing_candle(df):
    # ticker = 'TRG'
    # start_date = '2022-01-01'
    # end_date = '2024-03-06'

    # df = get_ticker_data_pak(ticker, start_date, end_date)

    df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

    float_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    for col in float_cols:
        df[col] = df[col].astype(float)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    df['signal'] = np.where((df.shift().Open > df.shift().Close) & 
                            (df.Open < df.Close) & 
                            (np.abs(df.Open - df.Close) > 
                             np.abs(df.shift().Open - df.shift().Close)), 1, 0)

    # df['x_val'] = (0.618 * (df['High'] - df['Low'])) + df['Low']
    # df['signal'] = np.where((df['Open'] < df['Close']) & 
    #                         (df['Open'] > df['x_val']), 1, 0)
    # df['next_signal'] = np.where(df['signal'].shift() == 1, -1, 0)
    # df['signal'] = np.where(df.next_signal == -1, -1, df.signal)
    # df.set_index('Date', inplace=True)
    # df = df.rename(columns={'signal':'predicted'})
    # df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'predicted']]
    df['next_signal'] = np.where((df['signal'].shift() == 1) & (df['Open'] < df['Close']), 1, 0)
    df['signal_rev'] = np.where((df['next_signal'] == 1) & (df['signal'].shift() == 1), 1, 0)
    # df['signal'] = np.where(df.next_signal == -1, -1, df.signal)
    # df.set_index('Date', inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal_rev']]
    df = df.rename(columns={'signal_rev':'signal'})


    return df

def ewm(df, period):
    df = df.sort_values(by='Date')
    df['ewm'] = df['Close'].ewm(span=period,min_periods=0,adjust=False,ignore_na=False).mean()
    return df

