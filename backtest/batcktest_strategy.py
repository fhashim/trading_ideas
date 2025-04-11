import backtrader as bt
from backtrader.feeds import PandasData
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime
import quantstats
from patterns.get_patterns import hammer
from get_data.read_data import get_tickers_list

stocks_list = ['FFC','ENGRO','SYS','EFERT','EPQL','EPCL','SEPL','HBL','POL','BAFL',
            'HMB','MTL','MARI','CHCC','UBL','ARPL','INIL','MCB','LOTCHEM','INDU',
            'JLICL','PCAL','HUBC','CHCC','PSO','FEROZ','DAWH','TRG','MEBL',
            'CYAN','EFUL','GADT','ISL','PKGS','AICL','FFBL','AKBL','ABOT','CEPB',
            'LCI','GGGL','REDCO','HUMNL','MRNS','TRIPF','AABS','SPEL','ACPL','OLPM',
            'NESTLE','LUCK','BAHL','COLG','APL','BATA','AGIL','EFUG','AGP','ZIL']

fitered_stocks = ['ENGRO','EFERT','EPQL','MTL','UBL','INDU','PCAL','HUBC','FEROZ','MEBL',
                'PKGS','AICL','FFBL','HUMNL','MRNS','AABS','SPEL','NESTLE', 'EFUG','AGP','ZIL']



df = hammer('AIRLINK', '2023-01-01', '2024-07-03')

# # Plot candlestick chart with markers
# fig, ax = mpf.plot(df, type='candle', style='yahoo', ylabel='Price', 
#                    ylabel_lower='Volume', addplot=mpf.make_addplot(df['predicted'], 
#                                                                    type='scatter', 
#                                                                    markersize=100, 
#                                                                    marker='^', 
#                                                                    color='g', 
#                                                                    panel=1))

# Show the plot
mpf.show()

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
        limdays = 7200,  # minutes after which the buy order expires (OCO)
        limdays2 = 7200, # minutes after which the sell order expires (OCO)
        hold = 10,  # number of timestamps after which we need to sell
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
                p1_buy = close * 1.05  # close * (1.0 - self.p.limit)
                p2_buy = p1_buy * 0.9 # p1_buy - 0.1 * close
                p3_buy = p1_buy * 1.1 # p1_buy + 0.05 * close

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


data = SignalData(dataname=df)

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
cerebro.broker.setcash(100000.0)

# Set the commission scheme
# cerebro.broker.addcommissioninfo(BinanceCommission(), name='binance')
# cerebro.broker.setcommission(
#         commission=0.0001, margin=None, mult=1.0)

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

# quantstats.reports.html(returns, output=r'Stats.html',
#                         title="TRG")
# print(stock) 
cerebro.plot()

tickers_list = get_tickers_list()
for ticker in tickers_list:
    df = hammer(ticker, '2024-03-04', '2024-03-05')
    # print(ticker)
    if df != -1:
        if df[df.predicted == 1].shape[0] > 0:
            print(ticker)
            print(df)