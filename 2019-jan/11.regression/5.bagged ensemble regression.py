import sys
path_to_scripts = 'E://'
sys.path.append(path_to_scripts)

from common_utils  import grid_search_plot_one_parameter_curves, \
    grid_search_plot_two_parameter_curves, get_best_model
from regression_utils import generate_nonlinear_synthetic_data_regression, generate_linear_synthetic_data_regression, \
    plot_model_2d_regression, plot_model_3d_regression, plot_data_2d_regression, plot_data_3d_regression, \
    grid_search_plot_models_regression, plot_coefficients_regression, \
    plot_target_and_transformed_target_regression, rmse
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble

#2d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

scoring = metrics.make_scorer(rmse, greater_is_better=False)
#scoring = metrics.make_scorer(metrics.r2_score)

rf_estimator = ensemble.RandomForestRegressor()
rf_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(rf_estimator, rf_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(rf_estimator, rf_grid, X_train, y_train )
final_model = get_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)

et_estimator = ensemble.ExtraTreesRegressor()
et_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(et_estimator, et_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(et_estimator, et_grid, X_train, y_train )
final_model = get_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)

#3d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

rf_estimator = ensemble.RandomForestRegressor()
rf_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(rf_estimator, rf_grid, X_train, y_train, scoring =  scoring)
final_model = get_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(final_model, X_train, y_train)

et_estimator = ensemble.ExtraTreesRegressor()
et_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(et_estimator, et_grid, X_train, y_train, scoring =  scoring)
final_model = get_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(final_model, X_train, y_train)
