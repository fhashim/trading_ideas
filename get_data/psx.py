import requests
import pandas as pd
import time as t
from get_data.cnxn import server_access

db_table = pd.DataFrame()

dates = pd.bdate_range('2024-03-08', '2024-04-17')

t0 = t.time()

for d in dates:
    try:
        date = d.strftime('%Y-%m-%d')

        url = 'https://dps.psx.com.pk/historical'

        payload = {'date': date}

        r = requests.post(url=url, data=payload,
                          headers={'X-Requested-With': 'XMLHttpRequest'})

        df = pd.read_html(r.text)

        data = df[0:1][0]

        df = data.drop(['CHANGE', 'CHANGE (%)', 'LDCP'], axis=1)

        df = df.rename(columns={'SYMBOL': 'Ticker', 'LDCP': 'LDCP',
                                'OPEN': 'Open', 'HIGH': 'High',
                                'LOW': 'Low', 'CLOSE': 'Close',
                                'VOLUME': 'Volume'})
        df['Date'] = date

        df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        db_table = pd.concat([df, db_table])

        engine = server_access()

        df.to_sql('tbl_historical_prices_pak', engine, if_exists='append', index=False)

    except:
        print(d)

t1 = t.time()
print('Exec time is: ', t1 - t0)
