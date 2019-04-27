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
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
from sklearn.preprocessing import PolynomialFeatures

scoring = metrics.make_scorer(rmse, greater_is_better=False)

#2d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

#uniform knn
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':list(range(1,10)) }
grid_search_plot_models_regression(knn_estimator, knn_grid, X_train, y_train)
grid_search_plot_one_parameter_curves(knn_estimator, knn_grid, X_train, y_train, scoring =  scoring)
knn_final_model = grid_search_best_model(knn_estimator, knn_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(knn_final_model, X_train, y_train)

#weighted knn
knn_estimator = neighbors.KNeighborsRegressor(weights='distance')
knn_grid = {'n_neighbors':list(range(1,10)) }
grid_search_plot_models_regression(knn_estimator, knn_grid, X_train, y_train)
grid_search_plot_one_parameter_curves(knn_estimator, knn_grid, X_train, y_train, scoring =  scoring)
knn_final_model = grid_search_best_model(knn_estimator, knn_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(knn_final_model, X_train, y_train)

#3d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

#uniform knn
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':list(range(1,10)) }
grid_search_plot_one_parameter_curves(knn_estimator, knn_grid, X_train, y_train, scoring =  scoring)
knn_final_model = grid_search_best_model(knn_estimator, knn_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(knn_final_model, X_train, y_train)
regression_performance(knn_final_model, X_test, y_test)

#weighted knn
knn_estimator = neighbors.KNeighborsRegressor(weights='distance')
knn_grid = {'n_neighbors':list(range(1,10)) }
grid_search_plot_one_parameter_curves(knn_estimator, knn_grid, X_train, y_train, scoring =  scoring)
knn_final_model = grid_search_best_model(knn_estimator, knn_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(knn_final_model, X_train, y_train)
regression_performance(knn_final_model, X_test, y_test)
