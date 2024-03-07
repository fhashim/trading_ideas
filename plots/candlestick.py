import mplfinance as mpf
import pandas as pd


def candlestick_chart(df):

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df = df.set_index('Date')
    float_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in float_cols:
        df[col] = df[col].astype(float)
    
    mpf.plot(df, type='candle', style='charles', title='Prices', ylabel='Price (PKR)')

