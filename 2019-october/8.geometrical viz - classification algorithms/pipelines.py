import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
from sklearn import preprocessing, linear_model, pipeline

X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
X, y = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=1000, noise=0.1)

cutils.plot_data_2d_classification(X, y)

stages = [
            ('features', preprocessing.PolynomialFeatures()),
            ('perceptron', linear_model.Perceptron(max_iter=1000))
        ]
perceptron_pipeline = pipeline.Pipeline(stages)
perceptron_pipeline_grid  = {'perceptron__penalty':['l1'], 'perceptron__alpha':[0, 0.1, 0.3, 0.5], 'features__degree':[2,3]}
pipeline_object = cutils.grid_search_best_model(perceptron_pipeline, perceptron_pipeline_grid, X, y)
final_estimator = pipeline_object.named_steps['perceptron']
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(pipeline_object, X, y)
