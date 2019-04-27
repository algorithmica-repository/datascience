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

##outliers in linear pattern
X, y = generate_linear_synthetic_data_regression(1000, 1, 1, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

#add outliers in features
X_train[::10] = 4
#add outliers in target
y_train[::10] = 250
plot_data_2d_regression(X_train, y_train)

# Fit linear model
lr_estimator = linear_model.LinearRegression()
lr_grid = {'normalize':[True, False]}
lr_model = grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(lr_model, X_train, y_train, title="LinearRegression")
regression_performance(lr_model, X_test, y_test)

# Robustly fit linear model with Huber Regressor algorithm
hr_estimator = linear_model.HuberRegressor()
hr_grid = { 'epsilon':[1.1, 1.2, 1.3, 1.5]}
hr_model = grid_search_best_model(hr_estimator, hr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(hr_model, X_train, y_train, title="HuberRegression")
regression_performance(hr_model, X_test, y_test)

# Robustly fit linear model with RANSAC algorithm
ransac_estimator = linear_model.RANSACRegressor()
ransac_grid = { 'max_trials':[100, 150] }
ransac_model = grid_search_best_model(ransac_estimator, ransac_grid, X_train, y_train, scoring = scoring )
inlier_mask = ransac_model.inlier_mask_
plot_model_2d_regression(ransac_model, X_train, y_train, title="RANSAC")
plot_data_2d_regression(X_train[inlier_mask], y_train[inlier_mask], new_window=False, color='yellowgreen')
plot_data_2d_regression(X_train[np.logical_not(inlier_mask)], y_train[np.logical_not(inlier_mask)], color='gold', new_window=False)
regression_performance(ransac_model, X_test, y_test)

##outliers in non-linear pattern
X, y = generate_nonlinear_synthetic_sine_data_regression(600)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train, x_limit=[-4,10], y_limit=[-2,10])

#add outliers in features
X_train[::10] = 9
#add outliers in target
y_train[::10] = 9

plot_data_2d_regression(X_train, y_train, x_limit=[-4,10], y_limit=[-2,10])

# Fit linear model
poly_lr_estimator = Pipeline([('features', PolynomialFeatures(3)) ,
                               ('estimator', linear_model.LinearRegression())]
                        )
poly_lr_grid = {'estimator__normalize':[True, False]}
poly_lr_model = grid_search_best_model(poly_lr_estimator, poly_lr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_lr_model, X_train, y_train, title="LinearRegression", x_limit=[-4,10], y_limit=[-2,10])
regression_performance(poly_lr_model, X_test, y_test)

# Robustly fit linear model with Huber Regressor algorithm
poly_hr_estimator = Pipeline([('features', PolynomialFeatures(3)) ,
                               ('estimator', linear_model.HuberRegressor())]
                        )
poly_hr_grid = { 'estimator__epsilon':[1.1, 1.2, 1.3, 1.5]}
poly_hr_model = grid_search_best_model(poly_hr_estimator, poly_hr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(poly_hr_model, X_train, y_train, title="HuberRegression", x_limit=[-4,10], y_limit=[-2,10])
regression_performance(poly_hr_model, X_test, y_test)

# Robustly fit linear model with RANSAC algorithm
poly_ransac_estimator = Pipeline([('features', PolynomialFeatures(3)) ,
                               ('estimator', linear_model.RANSACRegressor())]
                            )
poly_ransac_grid = { 'estimator__max_trials':[100, 150, 200] }
poly_ransac_model = grid_search_best_model(poly_ransac_estimator, poly_ransac_grid, X_train, y_train, scoring = scoring )
inlier_mask = poly_ransac_model.named_steps['estimator'].inlier_mask_
plot_model_2d_regression(poly_ransac_model, X_train, y_train, title="RANSAC", x_limit=[-4,10], y_limit=[-2,10])
plot_data_2d_regression(X_train[inlier_mask], y_train[inlier_mask], x_limit=[-4,10], y_limit=[-2,10], new_window=False, color='yellowgreen')
plot_data_2d_regression(X_train[np.logical_not(inlier_mask)], y_train[np.logical_not(inlier_mask)], new_window = False, color='gold', x_limit=[-4,10], y_limit=[-2,10])
regression_performance(poly_ransac_model, X_test, y_test)