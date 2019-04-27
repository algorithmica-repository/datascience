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

#linear pattern in 2d
X, y = generate_linear_synthetic_data_regression(n_samples=200, n_features=1, 
                                                 n_informative=1,
                                                 noise = 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_linear_model = grid_search_best_model(linear_estimator, linear_grid, X_train, y_train, scoring = scoring)
plot_model_2d_regression(final_linear_model, X_train, y_train)
regression_performance(final_linear_model, X_test, y_test)

#linear pattern in 3d
X, y = generate_linear_synthetic_data_regression(n_samples=200, n_features=2, 
                                                 n_informative=2,
                                                 noise = 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_linear_model = grid_search_best_model(linear_estimator, linear_grid, X_train, y_train, scoring=scoring)
plot_model_3d_regression(final_linear_model, X_train, y_train)
regression_performance(final_linear_model, X_test, y_test)
