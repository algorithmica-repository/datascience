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
X, y = generate_linear_synthetic_data_regression(n_samples=10000, n_features=2, 
                                                 n_informative=2,
                                                 noise = 100)
y = np.exp((y + abs(y.min())) / 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lin_ridge_estimator = linear_model.Ridge()
lin_ridge_grid = {'alpha':[0.1, 0.2, 0.3, 0.5]}
grid_search_plot_one_parameter_curves(lin_ridge_estimator, lin_ridge_grid, X_train, y_train, scoring =  scoring)
poly_ridge_final_model = grid_search_best_model(lin_ridge_estimator, lin_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(poly_ridge_final_model, X_train, y_train)
regression_performance(poly_ridge_final_model, X_test, y_test)

y_trans = np.log1p(y)
sns.distplot(y, hist=True)
sns.distplot(y_trans, hist=True)

X_train, X_test, y_train, y_test = train_test_split(X, y_trans, random_state = 1)

lin_ridge_estimator = linear_model.Ridge()
lin_ridge_grid = {'alpha':[0.1, 0.2, 0.3, 0.5]}
grid_search_plot_one_parameter_curves(lin_ridge_estimator, lin_ridge_grid, X_train, y_train, scoring =  scoring)
poly_ridge_final_model = grid_search_best_model(lin_ridge_estimator, lin_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(poly_ridge_final_model, X_train, y_train)
regression_performance(poly_ridge_final_model, X_test, y_test, np.expm1)