import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit
import statsmodels
print(statsmodels.__version__)

path = 'F:/'
df = pd.read_csv(os.path.join(path,'uk-deaths-from-bronchitis-emphys.csv'))
df.info()

df.columns = ['timestamp', 'y']
df.index = pd.to_datetime(df['timestamp'], format='%Y-%m')
df.drop('timestamp', axis=1, inplace=True)

tsp = TimeSeriesSplit(n_splits=3)

for train_ind, test_ind in tsp.split(df):
     print(train_ind, test_ind)
     
for train_ind, test_ind in tsp.split(df):
     train_data = df.iloc[train_ind]
     test_data = df.iloc[test_ind]
     print(train_data, test_data)