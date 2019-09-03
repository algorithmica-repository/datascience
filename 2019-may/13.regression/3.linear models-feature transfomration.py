import sys
path = 'E://utils'
sys.path.append(path)

import common_utils  as utils
import regression_utils as rutils
import kernel_utils as kutils
from sklearn import metrics, linear_model, svm, model_selection, preprocessing, pipeline

scoring = metrics.make_scorer(rutils.rmse, greater_is_better=False)

X, y = rutils.generate_nonlinear_synthetic_data_regression(n_samples=200, n_features=1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

stages = [
            ('features', preprocessing.PolynomialFeatures()),
            ('reg', linear_model.LinearRegression())
        ]
lr_pipeline = pipeline.Pipeline(stages)
lr_pipeline_grid  = {'reg__normalize':[True, False], 'features__degree':[2,3,5,10]}
pipeline_object = utils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train, scoring = scoring)
final_linear_model = pipeline_object.named_steps['reg']
print(final_linear_model.coef_)
print(final_linear_model.intercept_)
rutils.plot_model_2d_regression(pipeline_object, X_train, y_train)
rutils.regression_performance(pipeline_object, X_test, y_test)

stages = [
            ('features', kutils.GaussianFeatures() ),
            ('reg', linear_model.LinearRegression())
        ]
lr_pipeline = pipeline.Pipeline(stages)
lr_pipeline_grid  = {'reg__normalize':[True, False], 'features__n_centres':[15, 20, 30, 36] }
pipeline_object = utils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train, scoring = scoring)
final_linear_model = pipeline_object.named_steps['reg']
print(final_linear_model.coef_)
print(final_linear_model.intercept_)
rutils.plot_model_2d_regression(pipeline_object, X_train, y_train)
rutils.regression_performance(pipeline_object, X_test, y_test)

stages = [
            ('features', preprocessing.PolynomialFeatures()),
            ('reg', svm.LinearSVR())
        ]
svr_pipeline = pipeline.Pipeline(stages)
svr_pipeline_grid  = {'reg__C':[0.01, 0.1, 0.3, 0.5, 0.7, 1, 10, 20], 'features__degree':[2,3,4,5,6,8,10]}
pipeline_object = utils.grid_search_best_model(svr_pipeline, svr_pipeline_grid, X_train, y_train, scoring = scoring)
final_svr_model = pipeline_object.named_steps['reg']
print(final_svr_model.coef_)
print(final_svr_model.intercept_)
rutils.plot_model_2d_regression(pipeline_object, X_train, y_train)
rutils.regression_performance(pipeline_object, X_test, y_test)

stages = [
            ('features', kutils.GaussianFeatures()),
            ('reg', svm.LinearSVR())
        ]
svr_pipeline = pipeline.Pipeline(stages)
svr_pipeline_grid  = {'reg__C':[0.01, 0.1, 0.3, 0.5, 0.7, 1, 10, 20], 'features__n_centres':[15, 20, 30, 36] }
pipeline_object = utils.grid_search_best_model(svr_pipeline, svr_pipeline_grid, X_train, y_train, scoring = scoring)
final_svr_model = pipeline_object.named_steps['reg']
print(final_svr_model.coef_)
print(final_svr_model.intercept_)
rutils.plot_model_2d_regression(pipeline_object, X_train, y_train)
rutils.regression_performance(pipeline_object, X_test, y_test)