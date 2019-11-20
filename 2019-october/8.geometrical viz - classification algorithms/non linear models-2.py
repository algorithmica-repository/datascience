import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
from sklearn import model_selection, linear_model, svm, preprocessing, pipeline


#2-d classification pattern
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
X, y = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=1000, noise=0.1)

cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

#perceptron algorithm
stages = [
            ('features', preprocessing.PolynomialFeatures()),
            ('clf', linear_model.Perceptron(max_iter=1000))
        ]
perceptron_pipeline = pipeline.Pipeline(stages)
perceptron_pipeline_grid  = {'clf__penalty':['l1'], 'clf__alpha':[0, 0.1, 0.3, 0.5], 'features__degree':[2,3,5,10]}
pipeline_object = cutils.grid_search_best_model(perceptron_pipeline, perceptron_pipeline_grid, X_train, y_train)
final_estimator = pipeline_object.named_steps['clf']
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(pipeline_object, X_train, y_train)

#logistic regression algorithm
stages = [
            ('features', preprocessing.PolynomialFeatures()),
            ('clf', linear_model.LogisticRegression())
        ]
lr_pipeline = pipeline.Pipeline(stages)
lr_pipeline_grid  = {'clf__penalty':['l1'], 'clf__C':[0.01, 0.1, 0.3, 0.5], 'features__degree':[2,3,5,10]}
pipeline_object = cutils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train)
final_estimator = pipeline_object.named_steps['clf']
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(pipeline_object, X_train, y_train)

#linear svm algorithm
stages = [
            ('features', preprocessing.PolynomialFeatures()),
            ('clf', svm.LinearSVC())
        ]
svm_pipeline = pipeline.Pipeline(stages)
svm_pipeline_grid  = {'clf__penalty':['l2'], 'clf__C':[0.01, 0.1, 0.3, 0.5], 'features__degree':[2,3,5,10, 50, 100]}
pipeline_object = cutils.grid_search_best_model(svm_pipeline, svm_pipeline_grid, X_train, y_train)
final_estimator = pipeline_object.named_steps['clf']
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(pipeline_object, X_train, y_train)
