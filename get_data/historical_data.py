import pandas as pd
import investpy
from get_data.cnxn import server_access
import time as t


engine = server_access().connect()

# start date is not inclusive
inv_start_date = '17/08/2022' #'05/08/2022' 
inv_end_date = '02/10/2022'

# use inv_start_date + 1 day
start_date = '2022-08-18'
end_date = '2022-02-10'

# sleep time
sleep_time = 10

sql = '''

SELECT Ticker
FROM stocks.tbl_tickers
where Available = 1 and ticker not in (

select distinct(Ticker) ticker from stocks.tbl_historical_prices where Date between '{}' and '{}'
  
  )

'''.format(start_date, end_date)


tickers = pd.read_sql(sql, con=engine)

tickers_list = tickers.Ticker.to_list()

not_done = []
remove_ticker = []

t0 =t.time() 

for ticker in tickers_list:
    try:
        
        df = investpy.get_stock_historical_data(stock=ticker,
                                        country='United States',
                                        from_date = inv_start_date,
                                        to_date = inv_end_date)


        df['Ticker'] = ticker
        df = df.reset_index()
        df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[df.Date >= start_date]                                       

        t.sleep(sleep_time)
        df.to_sql('tbl_historical_prices', con=engine, index=False, if_exists='append')
    
    except IndexError:
        remove_ticker.append(ticker)
        print('Index Error {}'.format(ticker))
        t.sleep(sleep_time)

    except:
        not_done.append(ticker)
        print('Fail {}'.format(ticker))
        t.sleep(sleep_time)

t1 = t.time()
print('Exec time: {}'.format((t1-t0)/(60*60)))

df = investpy.get_stock_historical_data(stock='AAPL',
                                        country='United States',
                                        from_date='06/08/2022',
                                        to_date='07/08/2022')



# df = investpy.get_stock_historical_data(stock='AAPL',
#                                         country='United States',
#                                         from_date = inv_start_date,
#                                         to_date = inv_end_date)
