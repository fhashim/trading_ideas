import backtrader as bt
from backtrader.feeds import PandasData
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime
import quantstats
from data.live_data import live_data

df = pd.read_csv(r'data\btc_usdt_15min.csv')
# df = live_data('2024-01-01', '2024-01-05')
df['OpenDate'] = pd.to_datetime(df['OpenDate'])
df = df.sort_values(by='OpenDate')

cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols:
    df[col] = df[col].astype(float)
'''
0%, 38.2%, 50%, 61.8%, 78.6%, 100%
1) Close > Open 
2) normalized = (x -x_min / x_max - x_min)
3) x_val = 0.618 * (x_max - X_min) + x_min
4) O > x_val
'''
analysis_df = df[['OpenDate', 'Open', 'High', 'Low', 'Close', 'Volume']]
analysis_df['x_val'] = (0.618 * (analysis_df['High'] - analysis_df['Low'])) + analysis_df['Low']
analysis_df['signal'] = np.where((analysis_df['Open'] < analysis_df['Close']) & 
                                 (analysis_df['Open'] > analysis_df['x_val']), 1, 0)
analysis_df['next_signal'] = np.where(analysis_df['signal'].shift() == 1, -1, 0)
analysis_df['signal'] = np.where(analysis_df.next_signal == -1, -1, analysis_df.signal)
analysis_df.set_index('OpenDate', inplace=True)

plot_df = analysis_df[(analysis_df.index >= '2023-12-01')] #  & (analysis_df.signal.abs() == 1) 
# plot_df = analysis_df.copy()
# Plot candlestick chart with markers
fig, ax = mpf.plot(plot_df, type='candle', style='yahoo', ylabel='Price', 
                   ylabel_lower='Volume', addplot=mpf.make_addplot(plot_df['signal'], 
                                                                   type='scatter', 
                                                                   markersize=100, 
                                                                   marker='^', 
                                                                   color='g', 
                                                                   panel=1))

# Show the plot
mpf.show()

## Backtest ##
# Define the Backtrader strategyx

bt_df = plot_df.copy()
bt_df = bt_df.rename(columns={'signal': 'predicted'})
bt_df = bt_df[['Open', 'High', 'Low', 'Close', 'Volume', 'predicted']]
bt_df.index.column = 'Date'
# class to define the columns we will provide
class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']
    cols = OHLCV + ['predicted']
    # create lines
    lines = tuple(cols)
    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())

class BinanceCommission(bt.CommissionInfo):
    params = (
        ("commission", 10),  # 0.1% commission rate
        ("mult", 1.0),  # This is multiplied by the size of the trade
        ("margin", None),
        # ("commtype", bt.Commission.PerTrade),
        # ("stocklike", False),
        # ("commtype", bt.Commission.PerTrade),
        # ("percabs", True),
    )

# define backtesting strategy class
# Create a Stratey
class PatternsStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datapredicted = self.datas[0].predicted

        # To keep track of pending orders
        self.order = None

    params = dict(
        limit = 0.005,
        limdays = 30,  # minutes after which the buy order expires (OCO)
        limdays2 = 30, # minutes after which the sell order expires (OCO)
        hold = 4,  # number of timestamps after which we need to sell
        usebracket_buy = False,  # buy use order_target_size
        switchp1p2_buy = False,  # buy switch prices of order1 and order2
        usebracket_sell = False,  # buy use order_target_size
        switchp1p2_sell = False,  # buy switch prices of order1 and order2
        )

    def notify_order(self, order):
        print('{}: Order ref: {} / Type {} / Status {}'.format(
            self.data.datetime.datetime(0),  # .date(0)
            order.ref, 'Buy' * order.isbuy() or 'Sell',
            order.getstatusname()))

        if order.status == order.Completed:
            self.holdstart = len(self)

        if not order.alive() and order.ref in self.orefs:
            self.orefs.remove(order.ref)

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # self.size_buy = (
            #                 (self.broker.get_cash() * 0.95 / self.data.close[0])
            #         )

            if self.datapredicted == 1:
                close = self.data.close[0]
                p1_buy = close - 200 # close * (1.0 - self.p.limit)
                p2_buy = p1_buy - 100 # p1_buy - 0.1 * close
                p3_buy = p1_buy + 100 # p1_buy + 0.05 * close

                valid1_buy = datetime.timedelta(minutes=self.p.limdays)
                valid2_buy = valid3_buy = datetime.timedelta(minutes=self.p.limdays2)

                if self.p.switchp1p2_buy:
                    p1_buy, p2_buy = p2_buy, p1_buy
                    valid1_buy, valid2_buy = valid2_buy, valid1_buy

                if not self.p.usebracket_buy:
                    self.size_buy = (
                            (self.broker.get_cash() * 0.95 / p1_buy)
                    )
                    print(self.size_buy)

                    o1 = self.buy(exectype=bt.Order.Limit,
                                  price=p1_buy,
                                  valid=valid1_buy,
                                  size=self.size_buy,
                                  transmit=False)
                    
                    print('{}: Oref {} / Buy at {}'.format(
                        self.datetime.datetime(), o1.ref, p1_buy))

                    o2 = self.sell(exectype=bt.Order.Stop,
                                   price=p2_buy,
                                   valid=valid2_buy,
                                   size=self.size_buy,
                                   parent=o1,
                                   transmit=False)
                    print('{}: Oref {} / Sell Stop at {}'.format(
                        self.datetime.datetime(), o2.ref, p2_buy))

                    o3 = self.sell(exectype=bt.Order.Limit,
                                   price=p3_buy,
                                   valid=valid3_buy,
                                   size=self.size_buy,
                                   parent=o1,
                                   transmit=True)

                    print('{}: Oref {} / Sell Limit at {}'.format(
                        self.datetime.datetime(), o3.ref, p3_buy))

                    self.orefs = [o1.ref, o2.ref, o3.ref]
                
                else:
                    os = self.buy_bracket(
                        price=p1_buy, valid=valid1_buy,
                        stopprice=p2_buy, stopargs=dict(valid=valid2_buy),
                        limitprice=p3_buy, limitargs=dict(valid=valid3_buy), )

                    self.orefs = [o.ref for o in os]


        else:
            if (len(self) - self.holdstart) >= self.p.hold:
                self.close()


data = SignalData(dataname=bt_df)

# Create a cerebro entity
cerebro = bt.Cerebro()
cerebro.addanalyzer(bt.analyzers.PyFolio)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

# Add a strategy
cerebro.addstrategy(PatternsStrategy)

# Add the Data Feed to Cerebro
cerebro.adddata(data)

# Set our desired cash start
cerebro.broker.setcash(1000.0)

# Set the commission scheme
# cerebro.broker.addcommissioninfo(BinanceCommission(), name='binance')
cerebro.broker.setcommission(
        commission=0.0001, margin=None, mult=1.0)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
results = cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

strat = results[0]
pyfoliozer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

# To make it compatible with quantstats, remove the timezone awareness using the built-in tz_convert function.
returns.index = returns.index.tz_convert(None)

quantstats.reports.html(returns, output=r'Stats.html',
                        title="BTC")

cerebro.plot()

## With commission: 35.30
