from patterns.get_patterns import engulfing_candle
from get_data.read_data import get_ticker_data_pak
import mplfinance as mpf

ticker = 'TRG'
start_date = '2022-01-01'
end_date = '2024-03-06'
df = get_ticker_data_pak(ticker, start_date, end_date)
df = engulfing_candle(df)
df = df.set_index('Date')
# Plot candlestick chart with markers
fig, ax = mpf.plot(df, type='candle', style='yahoo', ylabel='Price', 
                    ylabel_lower='Volume', addplot=mpf.make_addplot(df['signal'], 
                                                                    type='scatter', 
                                                                    markersize=100, 
                                                                    marker='^', 
                                                                    color='g', 
                                                                    panel=1))

# Show the plot
mpf.show()


