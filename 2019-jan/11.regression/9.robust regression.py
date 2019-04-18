import sys
path_to_scripts = 'E://'
sys.path.append(path_to_scripts)

from common_utils  import get_continuous_features, get_categorical_features, \
    cast_cont_to_cat, get_categorical_imputers, get_continuous_imputers, \
    get_features_to_drop_on_missingdata, ohe, drop_features, get_scaler, \
    get_zero_variance_filter, feature_reduction_pca, feature_reduction_tsne, \
    plot_feature_importances, select_features, grid_search_plot_one_parameter_curves, \
    grid_search_plot_two_parameter_curves, get_best_model
from regression_utils import generate_nonlinear_synthetic_sine_data_regression, generate_nonlinear_synthetic_data_regression, generate_linear_synthetic_data_regression, \
    plot_model_2d_regression, plot_model_3d_regression, plot_data_2d_regression, plot_data_3d_regression, \
    grid_search_plot_models_regression, plot_coefficients_regression, \
    plot_target_and_transformed_target_regression, rmse
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import sklearn
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


X, y = generate_linear_synthetic_data_regression(1000, 1, 1, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

X_train[::2] = 4
y_train[::10] = 250

plot_data_2d_regression(X_train, y_train, new_window=True)

scoring = metrics.make_scorer(rmse, greater_is_better=False)

# Fit linear model
lr_estimator = linear_model.LinearRegression()
lr_grid = {'normalize':[True, False]}
lr_model,_,_ = get_best_model(lr_estimator, lr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(lr_model, X_train, y_train, title="LinearRegression", color_model='yellow', new_window=True)

# Robustly fit linear model with Huber Regressor algorithm
hr_estimator = linear_model.HuberRegressor()
hr_grid = { 'epsilon':[1.1, 1.2, 1.3, 1.5]}
hr_model, _ = get_best_model(hr_estimator, hr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(hr_model, X_train, y_train, title="HuberRegression", color_model='yellow', new_window=True)

# Robustly fit linear model with RANSAC algorithm
ransac_estimator = linear_model.RANSACRegressor()
ransac_grid = { 'max_trials':[100, 150] }
ransac_model, inlier_mask = get_best_model(ransac_estimator, ransac_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(ransac_model, X_train, y_train, title="RANSAC", color_model='red', new_window=True)
plot_data_2d_regression(X_train[inlier_mask], y_train[inlier_mask], color='yellowgreen',  new_window=True)
plot_data_2d_regression(X_train[np.logical_not(inlier_mask)], y_train[np.logical_not(inlier_mask)], color='gold')


X, y = generate_nonlinear_synthetic_sine_data_regression(600)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train, x_limit=[-4,10], y_limit=[-2,10])

X_train[::10] = 9
y_train[::10] = 9

plot_data_2d_regression(X_train, y_train, x_limit=[-4,10], y_limit=[-2,10])


# Fit linear model
poly_lr_estimator = Pipeline([('features', PolynomialFeatures(3)) ,
                               ('estimator', linear_model.LinearRegression())]
                        )
poly_lr_estimator.fit(X_train, y_train)
plot_model_2d_regression(poly_lr_estimator, X_train, y_train, title="LinearRegression", color_model='yellow', new_window=True, x_limit=[-4,10], y_limit=[-2,10])

# Robustly fit linear model with Huber Regressor algorithm
poly_hr_estimator = Pipeline([('features', PolynomialFeatures(3)) ,
                               ('estimator', linear_model.HuberRegressor())]
                        )
poly_hr_grid = { 'estimator__epsilon':[1.1, 1.2, 1.3, 1.5]}
poly_hr_model, _ = get_best_model(poly_hr_estimator, poly_hr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_hr_model, X_train, y_train, title="HuberRegression", color_model='yellow', new_window=True, x_limit=[-4,10], y_limit=[-2,10])


# Robustly fit linear model with RANSAC algorithm
poly_ransac_estimator = Pipeline([('features', PolynomialFeatures(3)) ,
                               ('estimator', linear_model.RANSACRegressor())]
                            )
poly_ransac_grid = { 'estimator__max_trials':[100, 150, 200] }
poly_ransac_model, inlier_mask = get_best_model(poly_ransac_estimator, poly_ransac_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_ransac_model, X_train, y_train, title="RANSAC", color_model='red', new_window=True, x_limit=[-4,10], y_limit=[-2,10])
plot_data_2d_regression(X_train[inlier_mask], y_train[inlier_mask], color='yellowgreen',  new_window=True, x_limit=[-4,10], y_limit=[-2,10])
plot_data_2d_regression(X_train[np.logical_not(inlier_mask)], y_train[np.logical_not(inlier_mask)], color='gold', x_limit=[-4,10], y_limit=[-2,10])
