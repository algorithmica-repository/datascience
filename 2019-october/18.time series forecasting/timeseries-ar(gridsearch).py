import math 
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa import ar_model
import matplotlib.pyplot as plt

def grid_search_best_model_timeseries_ar(df, grid, cv):
    best_param = None
    best_score = np.infty
    tsp = TimeSeriesSplit(n_splits=cv)
            
    for param in grid.get('lags'):
        scores = []
        for train_ind, test_ind in tsp.split(df):
            train_data = df.iloc[train_ind]
            test_data = df.iloc[test_ind]
            try:
                #print(train_data, test_data)
                estimator = ar_model.AutoReg(train_data, lags=param)
                res = estimator.fit() 
                #print(res.params)
                #get out of sample predictions with test data start and end
                pred = estimator.predict(res.params, test_data.index[0], test_data.index[-1])
                #print(pred)
                y_pred = pred.values.reshape(-1)
                y_test = test_data.values.reshape(-1)
                score = math.sqrt(metrics.mean_squared_error(y_test,  y_pred))
                scores.append(score)
            except:
                pass
        #print(scores)
        if len(scores) > 0  and np.mean(scores) < best_score :
            best_score = np.mean(scores)
            best_param = param
        
    if best_param is not None:
        estimator = ar_model.AutoReg(df, lags=best_param)
        res = estimator.fit()
        print("best parameters:" + str(best_param))
        print("validation rmse:" +  str(best_score))
        #get insample predictions with start and end indices
        predictions = estimator.predict(res.params, start=0, end=df.shape[0]-1 )
        y_pred = predictions.values.reshape(-1)
        y_train = df.values.reshape(-1)[best_param:]
        train_rmse = math.sqrt(metrics.mean_squared_error(y_train,  y_pred))
        print("train rmse:" + str(train_rmse))
        return estimator, res
    else:
        return None, None
    
path = 'F:/'
df = pd.read_csv(os.path.join(path,'uk-deaths-from-bronchitis-emphys.csv'))
df.info()

df.columns = ['timestamp', 'y']
df.index = pd.to_datetime(df['timestamp'], format='%Y-%m')
df.index.freq = 'MS'
df.drop('timestamp', axis=1, inplace=True)

#grid search and get final model with best parameters
ar_grid  = { 'lags':[2,3,4,5] }
estimator, res = grid_search_best_model_timeseries_ar(df, ar_grid, 3)
print(res.params)
print(res.summary())
plt.plot(df)
plt.figure()
res.resid.plot()

#get predictions for future(implicit intervals based on freq of train data)
start_index = pd.datetime(1980, 1, 1)
end_index = pd.datetime(1990, 12, 1)
pred = estimator.predict(res.params, start_index, end_index)
print(pred)

#get predictions for future(explicit intervals)
index = pd.date_range('1-1-1980', '12-1-1990', freq='MS')
pred = estimator.predict(res.params, index[0], index[-1])
print(pred)

plt.figure()
plt.plot(pred)
