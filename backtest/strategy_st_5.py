from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd

import datetime  # For datetime objects

# Import the backtrader platform
import backtrader as bt

# Import quantstats to generate report.
import quantstats

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

import numpy as np

from backtrader.feeds import PandasData

from talib import RSI

from joblib import dump

date_mode = 'Bi-annually'
date_mode = 'Yearly'

df = pd.read_csv(r'Data/BTC_TA.csv')

df.columns = ['Datetime', 'CloseDate', 'Open', 'High', 'Low', 'Close', 'Volume',
              'filled', 'avg', 'macd', 'macdsignal', 'macdhist', 'rsi', 'stochrsi',
              'rsi_min', 'rsi_max', 'returns', 'signal']

df['rsi'] = RSI(df['avg'], timeperiod=26)
df['rsi_min'] = df['rsi'].rolling(26).min()
df['rsi_max'] = df['rsi'].rolling(26).max()
df['stochrsi'] = (df['rsi'] - df['rsi_min']) / (df['rsi_max'] - df['rsi_min'])
df = df[df.index >= (df[df.count(axis=1) == df.shape[1]].index[0])]
df['Datetime'] = pd.to_datetime(df['Datetime'])  # , utc=True

train_data = df[['Datetime', 'macdhist', 'stochrsi', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal']]

train_data.set_index('Datetime', inplace=True)

train_data['category'] = np.where(train_data['signal'] == 1, 'buy',
                                  np.where(train_data['signal'] == -1, 'sell', 'hold')
                                  )

train_data = pd.get_dummies(train_data, columns=['category'])
train_data['stochrsi'].fillna(0.5, inplace=True)
model_data = train_data[(train_data.index >= '2020-01-01') & (train_data.index <= '2020-12-31')]

X_train = model_data[['macdhist', 'stochrsi']].values
y_train = model_data[['category_buy', 'category_hold', 'category_sell']].values

model = OneVsRestClassifier(LogisticRegression(multi_class='multinomial', solver='lbfgs')).fit(X_train, y_train)

X_test = train_data[(train_data.index >= '2018-01-01') & (train_data.index <= '2021-06-30')][
    ['macdhist', 'stochrsi']].values

dump(model, r'Strategy_3/model.pkl')

pred = pd.DataFrame(model.predict_proba(X_test),
                    columns=['pred_buy', 'pred_hold', 'pred_sell'],
                    index=pd.date_range('2018-01-01', '2021-06-30', freq='min')
                    )

pred = pred.join(train_data[['Open', 'High', 'Low', 'Close', 'Volume']])
pred['predicted'] = np.where(pred['pred_buy'] >= 0.7, 1,
                             np.where(pred['pred_sell'] >= 0.7, -1, 0)
                             )
pred = pred[['predicted', 'Open', 'High', 'Low', 'Close', 'Volume']]


# class to define the columns we will provide
class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    OHLCV = ['open', 'high', 'low', 'close', 'volume']
    cols = OHLCV + ['predicted']
    # create lines
    lines = tuple(cols)
    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())


# define backtesting strategy class
class MLStrategy(bt.Strategy):

    def notify_order(self, order):
        print('{}: Order ref: {} / Type {} / Status {}'.format(
            self.data.datetime.datetime(0),  # .date(0)
            order.ref, 'Buy' * order.isbuy() or 'Sell',
            order.getstatusname()))

        if order.status == order.Completed:
            self.holdstart = len(self)

        if not order.alive() and order.ref in self.orefs:
            self.orefs.remove(order.ref)

    def __init__(self):
        self.data_predicted = self.datas[0].predicted
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close

        self.orefs = list()
        self.size_buy = None
        self.size_sell = None

        if self.p.usebracket_buy:
            print('-' * 5, 'Using buy_bracket')
        if self.p.usebracket_sell:
            print('-' * 5, 'Using sell_bracket')

    params = dict(
        limit=0.005,
        limdays=3 * 60,
        limdays2=4 * 120,
        hold=4 * 120 + 1,
        usebracket_buy=False,  # buy use order_target_size
        switchp1p2_buy=False,  # buy switch prices of order1 and order2
        usebracket_sell=False,  # buy use order_target_size
        switchp1p2_sell=False,  # buy switch prices of order1 and order2

    )

    def next(self):
        if self.orefs:
            return  # pending orders do nothing

        # Buy Long
        if not self.position:
            if self.data_predicted == 1:
                close = self.data.close[0]
                p1_buy = close * (1.0 - self.p.limit)
                p2_buy = p1_buy - 0.1 * close
                p3_buy = p1_buy + 0.05 * close

                valid1_buy = datetime.timedelta(minutes=self.p.limdays)
                valid2_buy = valid3_buy = datetime.timedelta(minutes=self.p.limdays2)

                if self.p.switchp1p2_buy:
                    p1_buy, p2_buy = p2_buy, p1_buy
                    valid1_buy, valid2_buy = valid2_buy, valid1_buy

                if not self.p.usebracket_buy:
                    self.size_buy = (
                            (self.broker.get_cash() * 0.12 / p1_buy)
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

        elif self.position and self.broker.get_cash() >= 0.50 * 1000:
            if self.data_predicted == 1:
                close = self.data.close[0]
                p1_buy = close * (1.0 - self.p.limit)
                p2_buy = p1_buy - 0.1 * close
                p3_buy = p1_buy + 0.05 * close

                valid1_buy = datetime.timedelta(minutes=self.p.limdays)
                valid2_buy = valid3_buy = datetime.timedelta(minutes=self.p.limdays2)

                if self.p.switchp1p2_buy:
                    p1_buy, p2_buy = p2_buy, p1_buy
                    valid1_buy, valid2_buy = valid2_buy, valid1_buy

                if not self.p.usebracket_buy:
                    self.size_buy = (
                            (self.broker.get_cash() * 0.12 / p1_buy)
                    )
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


        else:  # in the market
            if (len(self) - self.holdstart) >= self.p.hold:
                self.close()


data = SignalData(dataname=pred)

cerebro = bt.Cerebro()
cerebro.broker.set_cash(1000)
cerebro.broker.setcommission(commission=0.00075)
cerebro.adddata(data, name='BTC')

cerebro.addstrategy(MLStrategy)
cerebro.addanalyzer(bt.analyzers.PyFolio)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

strat = results[0]
pyfoliozer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

# To make it compatible with quantstats, remove the timezone awareness using the built-in tz_convert function.
returns.index = returns.index.tz_convert(None)

quantstats.reports.html(returns, output=r'D:\crypto_trading\Strategy_5\{}\Stats.html'.format(date_mode),
                        title="BTC")

print('Analyzer:', strat.analyzers.ta.get_analysis()['won'])
print('Analyzer:', strat.analyzers.ta.get_analysis()['lost'])
print('Analyzer:', strat.analyzers.ta.get_analysis()['long'])
print('Analyzer:', strat.analyzers.ta.get_analysis()['short'])
print('SQN:', strat.analyzers.sqn.get_analysis())

cerebro.plot()

returns.to_csv(r'Strategy_5\{}\Returns.csv'.format(date_mode))
positions.to_csv(r'Strategy_5\{}\Positions.csv'.format(date_mode))
transactions.to_csv(r'Strategy_5\{}\Transactions.csv'.format(date_mode))
gross_lev.to_csv(r'Strategy_5\{}\Gross_leverage.csv'.format(date_mode))
