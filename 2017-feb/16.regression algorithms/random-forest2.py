import os
import pandas as pd
from sklearn import ensemble
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

rf_estimator = ensemble.RandomForestRegressor(random_state=10)
rf_grid = {'n_estimators':[3,4,5], 'max_features':[7,8]}

def rmse(y_true, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='mean_squared_error', cv=10, n_jobs=5)
rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_estimator_

n_tree = 0
for est in rf_grid_estimator.best_estimator_.estimators_: 
    dot_data = io.StringIO()
    tmp = est.tree_
    tree.export_graphviz(tmp, out_file = dot_data, feature_names = X_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf("regtree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1

