import math 
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa import ar_model

path = 'F:/'
df = pd.read_csv(os.path.join(path,'uk-deaths-from-bronchitis-emphys.csv'))
df.info()

df.columns = ['timestamp', 'y']
df.index = pd.to_datetime(df['timestamp'], format='%Y-%m').copy()
df.index.freq = 'MS'
df.drop('timestamp', axis=1, inplace=True)

#build model
estimator = ar_model.AutoReg(df, lags=5)
res = estimator.fit()
print(res.params)
print(res.model)
print(res.summary())

#using model
predictions = estimator.predict(res.params, start=0, end=df.shape[0]-1 )
print(predictions)
y_pred = predictions.values.reshape(-1)
y_train = df.values.reshape(-1)[5:]
train_rmse = math.sqrt(metrics.mean_squared_error(y_train,  y_pred))
print(train_rmse)
 
#evaluate model
tsp = TimeSeriesSplit(n_splits=3)
scores = []

for train_ind, test_ind in tsp.split(df):
     train_data = df.iloc[train_ind]
     test_data = df.iloc[test_ind]
     estimator =  ar_model.AutoReg(df, lags=5)
     res = estimator.fit()
     pred = estimator.predict(res.params,test_data.index[0], test_data.index[-1])
     y_pred = pred.values.reshape(-1)
     y_test = test_data.values.reshape(-1)
     score = math.sqrt(metrics.mean_squared_error(y_test,  y_pred))
     scores.append(score)
print(scores)
print(np.mean(scores))
