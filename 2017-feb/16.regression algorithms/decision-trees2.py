import os
import pandas as pd
import pydot
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
import io
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

dt_estimator = tree.DecisionTreeRegressor()
dt_grid = {'max_depth':[3,4,5]}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring='mean_squared_error', cv=10, n_jobs=5)

def rmse(y_true, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))
    
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring=metrics.make_scorer(rmse), cv=10, n_jobs=5)

#build model using entire train data
dt_grid_estimator.fit(X_train,y_train)

dt_grid_estimator.grid_scores_
dt_grid_estimator.best_estimator_

dot_data = io.StringIO() 
tree.export_graphviz(dt_grid_estimator.best_estimator_, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("dt1.pdf")