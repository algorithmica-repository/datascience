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

#2d nonlinear pattern
X, y = generate_nonlinear_synthetic_data_regression(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_regression(X_train, y_train)

#polynomial kernel linear regression
poly_kernel = KernelTransformer('poly', degree=40)
X_train1 = poly_kernel.fit_transform(X_train) 

kernel_lr_estimator = Pipeline([('features', KernelTransformer('poly')) ,
                                ('estimator', linear_model.LinearRegression())]
                            )
kernel_lr_grid = {'features__degree':[5, 10, 15, 20] }
grid_search_plot_models_2d_regression(kernel_lr_estimator, kernel_lr_grid, X_train, y_train )
grid_search_plot_one_parameter_curves(kernel_lr_estimator, kernel_lr_grid, X_train, y_train, scoring = scoring)
kernel_lr_final_model = grid_search_best_model(kernel_lr_estimator, kernel_lr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(kernel_lr_final_model, X_train, y_train)
regression_performance(kernel_lr_final_model, X_test, y_test)

#gaussian kernel linear regression
rbf_kernel = KernelTransformer('rbf', gamma=1)
X_train1 = rbf_kernel.fit_transform(X_train) 

kernel_lr_estimator = Pipeline([('features', KernelTransformer('rbf')) ,
                                ('estimator', linear_model.LinearRegression())]
                            )
kernel_lr_grid = {'features__gamma':[1, 2, 5, 10] }
grid_search_plot_models_2d_regression(kernel_lr_estimator, kernel_lr_grid, X_train, y_train )
grid_search_plot_one_parameter_curves(kernel_lr_estimator, kernel_lr_grid, X_train, y_train, scoring = scoring)
kernel_lr_final_model = grid_search_best_model(kernel_lr_estimator, kernel_lr_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(kernel_lr_final_model, X_train, y_train)
regression_performance(kernel_lr_final_model, X_test, y_test)

#l2-regularized kernel linear regression
kernel_ridge_estimator = kernel_ridge.KernelRidge(kernel='rbf')
kernel_ridge_grid = {'alpha':[0.1, 0.3, 0.5, 1.0], 'gamma':[0.1, 0.2] }
grid_search_plot_models_2d_regression(kernel_ridge_estimator, kernel_ridge_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(kernel_ridge_estimator, kernel_ridge_grid, X_train, y_train, scoring = scoring)
kernel_ridge_final_model = grid_search_best_model(kernel_ridge_estimator, kernel_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(kernel_ridge_final_model, X_train, y_train)
regression_performance(kernel_ridge_final_model, X_test, y_test)

kernel_ridge_estimator = kernel_ridge.KernelRidge(kernel='poly')
kernel_ridge_grid = {'alpha':[0.1, 0.3, 0.5, 1.0], 'degree':[2, 3, 4, 5] }
grid_search_plot_models_2d_regression(kernel_ridge_estimator, kernel_ridge_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(kernel_ridge_estimator, kernel_ridge_grid, X_train, y_train, scoring = scoring)
kernel_ridge_final_model = grid_search_best_model(kernel_ridge_estimator, kernel_ridge_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(kernel_ridge_final_model, X_train, y_train)
regression_performance(kernel_ridge_final_model, X_test, y_test)

#gaussian kernel linear svm regression
rbf_kernel = KernelTransformer('rbf', gamma=1)
X_train1 = rbf_kernel.fit_transform(X_train) 

kernel_lsvm_estimator = Pipeline([('features', KernelTransformer('rbf')) ,
                                ('estimator', svm.LinearSVR())]
                            )
kernel_lsvm_grid = {'features__gamma':[1, 2, 5, 10] }
grid_search_plot_models_2d_regression(kernel_lsvm_estimator, kernel_lsvm_grid, X_train, y_train )
grid_search_plot_one_parameter_curves(kernel_lsvm_estimator, kernel_lsvm_grid, X_train, y_train, scoring = scoring)
kernel_lsvm_final_model = grid_search_best_model(kernel_lsvm_estimator, kernel_lsvm_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(kernel_lsvm_final_model, X_train, y_train)
regression_performance(kernel_lsvm_final_model, X_test, y_test)

#regularized kernel svm regression
svm_estimator = svm.SVR(kernel='rbf')
svm_grid= {'C':[15, 30, 100], 'gamma':[0.1, 0.2], 'epsilon':[0.1, 0.2] }
grid_search_plot_two_parameter_curves(svm_estimator, svm_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_2d_regression(svm_estimator, svm_grid, X_train, y_train )
svm_final_model = grid_search_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(svm_final_model, X_train, y_train)
regression_performance(svm_final_model, X_test, y_test)

svm_estimator = svm.SVR(kernel='poly')
svm_grid = {'C':[1, 10, 50, 100], 'degree':[2, 3, 4, 5] }
grid_search_plot_two_parameter_curves(svm_estimator, svm_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_2d_regression(svm_estimator, svm_grid, X_train, y_train )
svm_final_model = grid_search_best_model(svm_estimator, svm_grid, X_train, y_train, scoring = scoring )
plot_model_2d_regression(svm_final_model, X_train, y_train)
regression_performance(svm_final_model, X_test, y_test)
