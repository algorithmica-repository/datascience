import sys
path = 'I://New Folder//utils'
sys.path.append(path)
import common_utils  as utils
import regression_utils as rutils
from sklearn import metrics, tree, neighbors, model_selection, ensemble

scoring = metrics.make_scorer(rutils.rmse, greater_is_better=False)

#linear pattern in 2d
X, y = rutils.generate_nonlinear_synthetic_data_regression(n_samples=200, n_features=1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

dt_estimator = tree.DecisionTreeRegressor()
dt_grid  = {'max_depth':list(range(1,9)) }
final_dt_model = utils.grid_search_best_model(dt_estimator, dt_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_dt_model, X_train, y_train)
rutils.regression_performance(final_dt_model, X_test, y_test)

knn_estimator = neighbors.KNeighborsRegressor()
knn_grid  = {'n_neighbors':list(range(1,21)), 'weights':['uniform', 'distance'] }
final_knn_model = utils.grid_search_best_model(knn_estimator,knn_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_knn_model, X_train, y_train)
rutils.regression_performance(final_knn_model, X_test, y_test)

rf_estimator = ensemble.RandomForestRegressor()
rf_grid = {'n_estimators':list(range(10,200,50)), 'max_depth':list(range(3,6))}
final_rf_model = utils.grid_search_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_rf_model, X_train, y_train)
rutils.regression_performance(final_rf_model, X_test, y_test)

et_estimator = ensemble.ExtraTreesRegressor()
et_grid = {'n_estimators':list(range(10,200,20)), 'max_depth':list(range(3,6))}
final_et_model = utils.grid_search_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_et_model, X_train, y_train)
rutils.regression_performance(final_et_model, X_test, y_test)

gb_estimator = ensemble.GradientBoostingRegressor()
gb_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,5)), 'learning_rate':[0.1,0.5]}
final_gb_model = utils.grid_search_best_model(gb_estimator, gb_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_gb_model, X_train, y_train)
rutils.regression_performance(final_gb_model, X_test, y_test)

