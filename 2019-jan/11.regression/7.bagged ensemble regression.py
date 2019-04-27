import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

from common_utils  import *
from regression_utils import *
from kernel_utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection, kernel_ridge
from sklearn.preprocessing import PolynomialFeatures

scoring = metrics.make_scorer(rmse, greater_is_better=False)

rf_estimator = ensemble.RandomForestRegressor()
rf_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(rf_estimator, rf_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(rf_estimator, rf_grid, X_train, y_train )
rf_final_model = grid_search_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(rf_final_model, X_train, y_train)
regression_performance(rf_final_model, X_test, y_test)

et_estimator = ensemble.ExtraTreesRegressor()
et_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(et_estimator, et_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(et_estimator, et_grid, X_train, y_train )
et_final_model = grid_search_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(et_final_model, X_train, y_train)
regression_performance(et_final_model, X_test, y_test)

#3d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

rf_estimator = ensemble.RandomForestRegressor()
rf_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(rf_estimator, rf_grid, X_train, y_train, scoring =  scoring)
et_final_model = grid_search_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(et_final_model, X_train, y_train)
regression_performance(et_final_model, X_test, y_test)

et_estimator = ensemble.ExtraTreesRegressor()
et_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(et_estimator, et_grid, X_train, y_train, scoring =  scoring)
et_final_model = grid_search_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(et_final_model, X_train, y_train)
regression_performance(et_final_model, X_test, y_test)