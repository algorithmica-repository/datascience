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
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline

#overfitting control - regularization of irrelvant/redundant features(higher dimensional) in polynomial regression
X, y = generate_nonlinear_synthetic_data_regression(40, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

scoring = metrics.make_scorer(rmse, greater_is_better=False)
#scoring = metrics.make_scorer(metrics.r2_score)

poly_estimator = Pipeline([('features', PolynomialFeatures()) ,
                          ('estimator', linear_model.LinearRegression())]
                        )
poly_grid = {'features__degree':[2,3,10,15,20] }
grid_search_plot_one_parameter_curves(poly_estimator, poly_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(poly_estimator, poly_grid, X_train, y_train )
final_model = get_best_model(poly_estimator, poly_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)


poly_ridge_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Ridge())]
                                )
poly_ridge_grid = {'features__degree':[2, 3, 9, 10], 
                   'estimator__alpha':[0.01, 0.1, 1, 10]
                   }
grid_search_plot_two_parameter_curves(poly_ridge_estimator, poly_ridge_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(poly_ridge_estimator, poly_ridge_grid, X_train, y_train )
final_model = get_best_model(poly_ridge_estimator, poly_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)
plot_coefficients_regression(poly_ridge_estimator, X_train, y_train)


poly_lasso_estimator = Pipeline([('features', PolynomialFeatures(15)) ,
                                 ('estimator', linear_model.Lasso())]
                                )
poly_lasso_grid = {'features__degree':[2, 3, 9, 10], 
                   'estimator__alpha':[0.001, 0.01, 0.1, 10] 
                   }
grid_search_plot_two_parameter_curves(poly_lasso_estimator, poly_lasso_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_regression(poly_lasso_estimator, poly_lasso_grid, X_train, y_train )
final_model = get_best_model(poly_lasso_estimator, poly_lasso_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(final_model, X_train, y_train)
plot_coefficients_regression(poly_lasso_estimator, X_train, y_train)

#overfitting control - regularization of irrelvant or redundant features in linear regression
X, y = generate_linear_synthetic_data_regression(n_samples=200, n_features=190, 
                                                 n_informative=190,
                                                 noise = 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

linear_estimator = linear_model.LinearRegression()
linear_grid = {'normalize':[True, False]}
final_model = get_best_model(linear_estimator, linear_grid, X_train, y_train, scoring = scoring )

ridge_estimator = linear_model.Ridge()
ridge_grid = { 'alpha': [0.01, 0.1, 0.2, 0.5, 0.9, 1, 10, 20, 30] }
grid_search_plot_one_parameter_curves(ridge_estimator, ridge_grid, X_train, y_train, scoring =  scoring)
final_model = get_best_model(ridge_estimator, ridge_grid, X_train, y_train, scoring = scoring )

lasso_estimator = linear_model.Lasso()
lasso_grid = { 'alpha': [0.01, 0.1, 0.2, 0.5, 0.9, 1, 5, 10, 15] }
grid_search_plot_one_parameter_curves(lasso_estimator, lasso_grid, X_train, y_train, scoring =  scoring)
final_model = get_best_model(lasso_estimator, lasso_grid, X_train, y_train, scoring = scoring )