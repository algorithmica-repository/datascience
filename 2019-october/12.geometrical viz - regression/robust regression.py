import sys
path = 'I:/New Folder/utils'
sys.path.append(path)
import common_utils  as utils
import regression_utils as rutils
from sklearn import metrics, linear_model, model_selection
import numpy as np

scoring = metrics.make_scorer(rutils.rmse, greater_is_better=False)

##outliers in linear pattern
X, y = rutils.generate_linear_synthetic_data_regression(1000, 1, 1, 10)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

#add outliers in features
X_train[::10] = 4
#add outliers in target
y_train[::10] = 250
rutils.plot_data_2d_regression(X_train, y_train)

# Fit linear model
lr_estimator = linear_model.LinearRegression()
lr_grid = {'normalize':[False]}
lr_model = utils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring = scoring )
rutils.plot_model_2d_regression(lr_model, X_train, y_train, title="LinearRegression")
rutils.regression_performance(lr_model, X_test, y_test)

# Robustly fit linear model with Huber Regressor algorithm
hr_estimator = linear_model.HuberRegressor()
hr_grid = { 'epsilon':[1.1, 1.2, 1.3, 1.5]}
hr_model = utils.grid_search_best_model(hr_estimator, hr_grid, X_train, y_train, scoring = scoring )
rutils.plot_model_2d_regression(hr_model, X_train, y_train, title="HuberRegression")
rutils.regression_performance(hr_model, X_test, y_test)

# Robustly fit linear model with RANSAC algorithm
ransac_estimator = linear_model.RANSACRegressor()
ransac_grid = { 'max_trials':[100, 150] }
ransac_model = utils.grid_search_best_model(ransac_estimator, ransac_grid, X_train, y_train, scoring = scoring )
inlier_mask = ransac_model.inlier_mask_
rutils.plot_model_2d_regression(ransac_model, X_train, y_train, title="RANSAC")
rutils.plot_data_2d_regression(X_train[inlier_mask], y_train[inlier_mask], new_window=False, color='yellowgreen')
rutils.plot_data_2d_regression(X_train[np.logical_not(inlier_mask)], y_train[np.logical_not(inlier_mask)], color='gold', new_window=False)
rutils.regression_performance(ransac_model, X_test, y_test)
