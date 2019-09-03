import sys
path = 'E://utils'
sys.path.append(path)

import common_utils  as utils
import regression_utils as rutils
from sklearn import metrics, linear_model, svm, model_selection

scoring = metrics.make_scorer(rutils.rmse, greater_is_better=False)

#linear pattern in 2d
X, y = rutils.generate_linear_synthetic_data_regression(n_samples=200, n_features=1, 
                                                 n_informative=1,
                                                 noise = 100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_linear_model = utils.grid_search_best_model(linear_estimator, linear_grid, X_train, y_train, scoring = scoring)
print(final_linear_model.coef_)
print(final_linear_model.intercept_)
rutils.plot_model_2d_regression(final_linear_model, X_train, y_train)
rutils.regression_performance(final_linear_model, X_test, y_test)

svm_estimator = svm.LinearSVR()
svm_grid = {'C':[0.1, 0.3, 0.5, 0.7, 1, 10] }
final_svm_model = utils.grid_search_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring)
print(final_svm_model.coef_)
print(final_svm_model.intercept_)
rutils.plot_model_2d_regression(final_svm_model, X_train, y_train)
rutils.regression_performance(final_svm_model, X_test, y_test)

#linear pattern in 3d
X, y = rutils.generate_linear_synthetic_data_regression(n_samples=200, n_features=2, 
                                                 n_informative=2,
                                                 noise = 10)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_3d_regression(X_train, y_train)

linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_linear_model = utils.grid_search_best_model(linear_estimator, linear_grid, X_train, y_train, scoring=scoring)
print(final_linear_model.coef_)
print(final_linear_model.intercept_)
rutils.plot_model_3d_regression(final_linear_model, X_train, y_train)
rutils.regression_performance(final_linear_model, X_test, y_test)

svm_estimator = svm.LinearSVR()
svm_grid = {'C':[0.1, 0.3, 0.5, 0.7, 1, 10] }
final_svm_model = utils.grid_search_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring)
print(final_svm_model.coef_)
print(final_svm_model.intercept_)
rutils.plot_model_3d_regression(final_svm_model, X_train, y_train)
rutils.regression_performance(final_svm_model, X_test, y_test)


