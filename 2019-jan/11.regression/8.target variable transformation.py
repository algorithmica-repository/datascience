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

X, y = generate_linear_synthetic_data_regression(n_samples=10000, n_features=2, 
                                                 n_informative=2,
                                                 noise = 100)
y = np.exp((y + abs(y.min())) / 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lin_ridge_estimator = linear_model.Ridge()
lin_ridge_estimator.fit(X_train, y_train)
y_pred = lin_ridge_estimator.predict(X_test)
plot_data_2d_regression(y_test.reshape(-1,1), y_pred.reshape(-1,1))

y_trans = np.log1p(y)
plot_target_and_transformed_target_regression(y, y_trans)

X_train, X_test, y_train, y_test = train_test_split(X, y_trans, random_state = 1)
lin_ridge_estimator.fit(X_train, y_train)
y_pred = lin_ridge_estimator.predict(X_test)
plot_data_2d_regression(np.expm1(y_test).reshape(-1,1), np.expm1(y_pred).reshape(-1,1) )