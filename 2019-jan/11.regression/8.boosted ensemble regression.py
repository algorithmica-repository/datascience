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

#2d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

dt_estimator = tree.DecisionTreeRegressor()
ada_estimator = ensemble.AdaBoostRegressor(base_estimator=dt_estimator)
ada_grid = {'n_estimators':list(range(100,300,100)), 'base_estimator__max_depth':list(range(1,4)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_2d_regression(ada_estimator, ada_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(gb_estimator, gb_grid, X_train, y_train, scoring =  scoring)
ada_final_model = grid_search_best_model(ada_estimator, ada_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(ada_final_model, X_train, y_train)
regression_performance(ada_final_model, X_test, y_test)

gb_estimator = ensemble.GradientBoostingRegressor()
gb_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,5)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_2d_regression(gb_estimator, gb_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(gb_estimator, gb_grid, X_train, y_train, scoring =  scoring)
gb_final_model = grid_search_best_model(gb_estimator, gb_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(gb_final_model, X_train, y_train)
regression_performance(gb_final_model, X_test, y_test)

#3d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

dt_estimator = tree.DecisionTreeRegressor()
ada_estimator = ensemble.AdaBoostRegressor(base_estimator=dt_estimator)
ada_grid = {'n_estimators':list(range(10,100,40)), 'base_estimator__max_depth':list(range(3,5)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_3d_regression(ada_estimator, ada_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(gb_estimator, gb_grid, X_train, y_train, scoring =  scoring)
ada_final_model = grid_search_best_model(ada_estimator, ada_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(ada_final_model, X_train, y_train)
regression_performance(ada_final_model, X_test, y_test)

gb_estimator = ensemble.GradientBoostingRegressor()
gb_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,5)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_3d_regression(gb_estimator, gb_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(gb_estimator, gb_grid, X_train, y_train, scoring =  scoring)
gb_final_model = grid_search_best_model(gb_estimator, gb_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(gb_final_model, X_train, y_train)
regression_performance(gb_final_model, X_test, y_test)
