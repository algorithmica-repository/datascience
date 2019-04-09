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
from sklearn import metrics
from sklearn import svm

#2d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

scoring = metrics.make_scorer(rmse, greater_is_better=False)
#scoring = metrics.make_scorer(metrics.r2_score)

svm_estimator = svm.SVR(kernel='rbf')
svm_grid= {'C':[15, 30, 100], 'gamma':[0.1, 0.2], 'epsilon':[0.1, 0.2] }
grid_search_plot_two_parameter_curves(svm_estimator, svm_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(svm_estimator, svm_grid, X_train, y_train )
final_model = get_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)

svm_estimator = svm.SVR(kernel='poly')
svm_grid = {'C':[1, 10, 50, 100], 'degree':[2, 3, 4, 5] }
grid_search_plot_two_parameter_curves(svm_estimator, svm_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(svm_estimator, svm_grid, X_train, y_train )
final_model = get_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)


#3d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(40, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

svm_estimator = svm.SVR(kernel='rbf')
svm_grid= {'C':[15, 30, 100], 'gamma':[0.1, 0.2], 'epsilon':[0.1, 0.2] }
grid_search_plot_two_parameter_curves(svm_estimator, svm_grid, X_train, y_train, scoring =  scoring)
final_model = get_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(final_model, X_train, y_train)

svm_estimator = svm.SVR(kernel='poly')
svm_grid = {'C':[1, 10, 50, 100], 'degree':[2, 3, 4, 5] }
grid_search_plot_two_parameter_curves(svm_estimator, svm_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(svm_estimator, svm_grid, X_train, y_train )
final_model = get_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(final_model, X_train, y_train)
