import sys
path = 'I://New Folder//utils'
sys.path.append(path)
import common_utils  as utils
import kernel_utils as kutils
import regression_utils as rutils
from sklearn import metrics, model_selection,  linear_model, preprocessing, pipeline, kernel_ridge, svm

scoring = metrics.make_scorer(rutils.rmse, greater_is_better=False)

#linear pattern in 2d
X, y = rutils.generate_nonlinear_synthetic_data_regression(n_samples=200, n_features=1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

lr_pipeline = pipeline.Pipeline([
            ('features', preprocessing.PolynomialFeatures()),
            ('reg', linear_model.LinearRegression())
        ])
lr_pipeline_grid  = {'features__degree':[2,3,5,10]}
pipeline_object = utils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train, scoring = scoring)
final_linear_model = pipeline_object.named_steps['reg']
print(final_linear_model.coef_)
print(final_linear_model.intercept_)
rutils.plot_model_2d_regression(pipeline_object, X_train, y_train)
rutils.regression_performance(pipeline_object, X_test, y_test)

lr_pipeline  =  pipeline.Pipeline([
            ('features', kutils.GaussianFeatures() ),
            ('reg', linear_model.LinearRegression())
        ])
lr_pipeline_grid  = {'features__n_centres':[15, 20, 30, 36] }
pipeline_object = utils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train, scoring = scoring)
final_linear_model = pipeline_object.named_steps['reg']
print(final_linear_model.coef_)
print(final_linear_model.intercept_)
rutils.plot_model_2d_regression(pipeline_object, X_train, y_train)
rutils.regression_performance(pipeline_object, X_test, y_test)

kernel_lr = kernel_ridge.KernelRidge(kernel="rbf")
kernel_lr_grid  = {'alpha':[0.0001, 0.01, 0.05, 0.2, 0.5, 1], 'gamma':[0.01, 0.1, 1, 2, 3, 4, 5, 10]}
final_kernel_lr_model = utils.grid_search_best_model(kernel_lr, kernel_lr_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_kernel_lr_model, X_train, y_train)
rutils.regression_performance(final_kernel_lr_model, X_test, y_test)

kernel_svm = svm.SVR(kernel="rbf")
kernel_svm_grid  = {'C':[0.2, 0.5, 10, 20, 50, 100], 'gamma':[0.01, 0.1, 1, 2, 3, 4, 5, 10]}
final_kernel_svm_model = utils.grid_search_best_model(kernel_svm, kernel_svm_grid, X_train, y_train, scoring = scoring)
rutils.plot_model_2d_regression(final_kernel_svm_model, X_train, y_train)
rutils.regression_performance(final_kernel_svm_model, X_test, y_test)
