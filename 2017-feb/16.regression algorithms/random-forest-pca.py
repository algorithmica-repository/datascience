import os
import pandas as pd
from sklearn import ensemble
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
import math
from sklearn import preprocessing
from sklearn import decomposition
from sklearn_pandas import DataFrameMapper
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

mapper = DataFrameMapper([(restaurant_train1.columns, preprocessing.StandardScaler())])
scaled_features = mapper.fit_transform(restaurant_train1)
restaurant_train2 = pd.DataFrame(scaled_features, columns=restaurant_train1.columns)

pca = decomposition.PCA(n_components=30)
pca.fit(restaurant_train2)
print(pca.explained_variance_ratio_.cumsum())
restaurant_train3 = pca.transform(restaurant_train2)

X_train = restaurant_train3
y_train = restaurant_train['revenue']

rf_estimator = ensemble.RandomForestRegressor(random_state=10)
rf_grid = {'n_estimators':[3,4,5], 'max_features':[7,8]}

def rmse(y_true, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='mean_squared_error', cv=10, n_jobs=5)
rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_

