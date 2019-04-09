import sys
path_to_scripts = 'G://'
sys.path.append(path_to_scripts)

from common_utils  import grid_search_plot_one_parameter_curves, \
    grid_search_plot_two_parameter_curves, get_best_model
from regression_utils import generate_nonlinear_synthetic_data_regression, generate_linear_synthetic_data_regression, \
    plot_model_2d_regression, plot_model_3d_regression, plot_data_2d_regression, plot_data_3d_regression, \
    grid_search_plot_models_regression, plot_coefficients_regression, \
    plot_target_and_transformed_target_regression, rmse
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

scoring = metrics.make_scorer(rmse, greater_is_better=False)

#linear pattern in 2d
X, y = generate_linear_synthetic_data_regression(n_samples=200, n_features=1, 
                                                 n_informative=1,
                                                 noise = 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)
linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize': [True, False]}
final_model = get_best_model(linear_estimator, linear_grid, X_train, y_train, scoring = scoring)
plot_model_2d_regression(final_model, X_train, y_train)

#linear pattern in 3d
X, y = generate_linear_synthetic_data_regression(n_samples=200, n_features=2, 
                                                 n_informative=2,
                                                 noise = 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)
final_model = get_best_model(linear_estimator, linear_grid, X_train, y_train, scoring=scoring)
plot_model_3d_regression(final_model, X_train, y_train)