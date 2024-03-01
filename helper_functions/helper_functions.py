import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def median(x):
    return np.median(x)

def days_in_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 366
    return 365

def day_of_year_cyclic(df):
    df['doy_sin_time'] = np.sin(2*np.pi*df.seconds/df['days_in_year'])
    df['doy_cos_time'] = np.cos(2*np.pi*df.seconds/df['days_in_year'])
    return df

df = pd.read_csv('btc_usdt.csv')

cols = ['OpenDate', 'CloseDate', 'Open', 'High', 'Low', 'Close',
       'Volume']

btc = df[cols]

df['CloseHourDate'] = pd.to_datetime(df['CloseEpoch'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
df['OpenHourDate'] = pd.to_datetime(df['OpenEpoch'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
df['OpenHourDate'] = pd.to_datetime(df['OpenHourDate'])
df['CloseHourDate'] = pd.to_datetime(df['CloseHourDate'])

btc_hr = pd.DataFrame(pd.date_range('2022-01-01 00:00:59', '2023-03-31 11:59:59', freq='1H'), columns=['CloseHourDate'])

btc_hr = btc_hr.set_index('CloseHourDate').join(df.set_index(['CloseHourDate']))

btc_hr = btc_hr[['Open', 'High', 'Low', 'Close', 'Volume']]

btc_hr['CloseChg'] = btc_hr['Close'].pct_change()

btc_hr = btc_hr[~(btc_hr.CloseChg.isna())]

btc_hr['hour'] = btc_hr.index.hour

btc_hr = btc_hr.reset_index()

# day of year cyclic
btc_hr['days_in_year'] = btc_hr.CloseHourDate.dt.year.apply(days_in_year)
btc_hr['doy_sin_time'] = np.sin(2*np.pi*btc_hr.CloseHourDate.dt.day/btc_hr['days_in_year'])
btc_hr['doy_cos_time'] = np.cos(2*np.pi*btc_hr.CloseHourDate.dt.day/btc_hr['days_in_year'])

# month of year cyclic
btc_hr['moy_sin_time'] = np.sin(2*np.pi*btc_hr.CloseHourDate.dt.month/12)
btc_hr['moy_cos_time'] = np.cos(2*np.pi*btc_hr.CloseHourDate.dt.month/12)

# day of month cyclic
btc_hr['dom_sin_time'] = np.sin(2*np.pi*btc_hr.CloseHourDate.dt.month/btc_hr.CloseHourDate.days_in_month)
btc_hr['dom_cos_time'] = np.cos(2*np.pi*btc_hr.CloseHourDate.dt.month/btc_hr.CloseHourDate.days_in_month)

# day of week cyclic
btc_hr['dow_sin_time'] = np.sin(2*np.pi*btc_hr.CloseHourDate.dt.dayofweek/7)
btc_hr['dow_cos_time'] = np.cos(2*np.pi*btc_hr.CloseHourDate.dt.dayofweek/7)

# hour of day cyclic
btc_hr['hod_sin_time'] = np.sin(2*np.pi*btc_hr.CloseHourDate.dt.hour/24)
btc_hr['hod_cos_time'] = np.cos(2*np.pi*btc_hr.CloseHourDate.dt.hour/24)


# Prepare the features for clustering (select relevant columns)
X = btc_hr[['hour', 'CloseChg']]

# Apply the elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares (WCSS)
max_clusters = 10  # Maximum number of clusters to test

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, max_clusters + 1), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow graph, choose the optimal number of clusters (k)
optimal_k = 12  # Replace with the value determined from the graph

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to the original dataset
btc_hr['Cluster'] = y_kmeans

# You can now analyze the dataset with the added cluster information
print(btc_hr.head())

btc_hr_grouped = btc_hr.groupby('Cluster').agg({'hour':['min', 'max'], 
                         'CloseChg':['min', 'max', 'mean', median, np.std]})

btc_hr_grouped.columns = ['_'.join(col).strip() for col in btc_hr_grouped.columns.values]

btc_hr_grouped.sort_values(by='CloseChg_max', ascending=False)