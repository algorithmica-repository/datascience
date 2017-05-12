import os
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
import math
#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:\\revenue-prediction")

restaurant_train = pd.read_csv("train.csv")
restaurant_train.shape
restaurant_train.info()

restaurant_train1 = pd.get_dummies(restaurant_train, columns=['City Group', 'Type'])
restaurant_train1.shape
restaurant_train1.info()
restaurant_train1.drop(['Id','Open Date','City','revenue'], axis=1, inplace=True)

X_train = restaurant_train1
y_train = restaurant_train['revenue']

rf_estimator = ensemble.RandomForestRegressor(random_state=10)
rf_grid = {'n_estimators':[250,300,400,500], 'max_features':[7,8]}

def rmse(y_true, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='mean_squared_error', cv=10, n_jobs=8)
rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_

math.sqrt(6297309417510)

