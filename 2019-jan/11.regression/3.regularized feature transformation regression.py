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

#applying polynomial basis transformer
poly_transformer = PolynomialFeatures(15)
X_train1 = poly_transformer.fit_transform(X_train) 

#polynomial linear regression
poly_lr_estimator = Pipeline([('features', PolynomialFeatures()) ,
                              ('estimator', linear_model.LinearRegression())]
                            )
poly_lr_grid = {'features__degree':[1, 10, 20, 40] }
grid_search_plot_models_2d_regression(poly_lr_estimator, poly_lr_grid, X_train, y_train )
grid_search_plot_one_parameter_curves(poly_lr_estimator, poly_lr_grid, X_train, y_train, scoring = scoring)
poly_lr_final_model = grid_search_best_model(poly_lr_estimator, poly_lr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_lr_final_model, X_train, y_train)
regression_performance(poly_lr_final_model, X_test, y_test)

#regularized ridge polynomial regression
poly_ridge_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Ridge())]
                                )
poly_ridge_grid = {'features__degree':[2, 3, 9, 10], 
                   'estimator__alpha':[0.01, 0.1, 1, 10]
                   }
grid_search_plot_models_2d_regression(poly_ridge_estimator, poly_ridge_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(poly_ridge_estimator, poly_ridge_grid, X_train, y_train, scoring =  scoring)
poly_ridge_final_model = grid_search_best_model(poly_ridge_estimator, poly_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_ridge_final_model, X_train, y_train)
regression_performance(poly_lr_final_model, X_test, y_test)
plot_coefficients_regression(poly_ridge_estimator, X_train, y_train)

#regularized lasso polynomial regression
poly_lasso_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Lasso())]
                                )
poly_lasso_grid = {'features__degree':[2, 3, 9, 10], 
                   'estimator__alpha':[0.001, 0.01, 0.1, 10] 
                   }
grid_search_plot_models_2d_regression(poly_lasso_estimator, poly_lasso_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(poly_lasso_estimator, poly_lasso_grid, X_train, y_train, scoring =  scoring)
poly_lasso_final_model = grid_search_best_model(poly_lasso_estimator, poly_lasso_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_lasso_final_model, X_train, y_train)
regression_performance(poly_lasso_final_model, X_test, y_test)
plot_coefficients_regression(poly_lasso_estimator, X_train, y_train)

#3d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(100, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_3d_regression(X_train, y_train)

#applying polynomial basis transformer
poly_transformer = PolynomialFeatures(15)
X_train1 = poly_transformer.fit_transform(X_train) 

poly_lr_estimator = Pipeline([('features', PolynomialFeatures()) ,
                              ('estimator', linear_model.LinearRegression())]
                            )
poly_lr_grid = {'features__degree':[1, 10, 20, 40] }
grid_search_plot_models_3d_regression(poly_lr_estimator, poly_lr_grid, X_train, y_train )
grid_search_plot_one_parameter_curves(poly_lr_estimator, poly_lr_grid, X_train, y_train, scoring = scoring)
poly_lr_final_model = grid_search_best_model(poly_lr_estimator, poly_lr_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(poly_lr_final_model, X_train, y_train)
regression_performance(poly_lr_final_model, X_test, y_test)

#regularized ridge polynomial regression
poly_ridge_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Ridge())]
                                )
poly_ridge_grid = {'features__degree':[2, 3, 9, 10], 
                   'estimator__alpha':[0.01, 0.1, 1, 10]
                   }
grid_search_plot_models_3d_regression(poly_ridge_estimator, poly_ridge_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(poly_ridge_estimator, poly_ridge_grid, X_train, y_train, scoring =  scoring)
poly_ridge_final_model = grid_search_best_model(poly_ridge_estimator, poly_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(poly_ridge_final_model, X_train, y_train)
regression_performance(poly_ridge_final_model, X_test, y_test)
plot_coefficients_regression(poly_ridge_estimator, X_train, y_train)

#regularized lasso polynomial regression
poly_lasso_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Lasso())]
                                )
poly_lasso_grid = {'features__degree':[2, 3, 9, 10], 
                   'estimator__alpha':[0.001, 0.01, 0.1, 10] 
                   }
grid_search_plot_models_3d_regression(poly_lasso_estimator, poly_lasso_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(poly_lasso_estimator, poly_lasso_grid, X_train, y_train, scoring =  scoring)
poly_lasso_final_model = grid_search_best_model(poly_lasso_estimator, poly_lasso_grid, X_train, y_train, scoring = scoring )
plot_model_3d_regression(poly_lasso_final_model, X_train, y_train)
regression_performance(poly_lasso_final_model, X_test, y_test)
plot_coefficients_regression(poly_lasso_estimator, X_train, y_train)
