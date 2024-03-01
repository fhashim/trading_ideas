import pandas as pd
import investpy
from get_data.cnxn import server_access
import time as t


engine = server_access().connect()


# index_list = ['S&P500', '']
# df = investpy.get_index_historical_data(index='US500',
#                                         country='United States',
#                                         from_date='06/07/2022',
#                                         to_date='07/07/2022')


us_df = pd.DataFrame(investpy.indices.get_indices_dict(country='United States')) 
china_df =  pd.DataFrame(investpy.indices.get_indices_dict(country='China')) 
df = pd.concat([us_df, china_df])
df.to_csv('index.csv', index=False)

# df = pd.DataFrame(investpy.indices.get_indices_list(country='United States'))