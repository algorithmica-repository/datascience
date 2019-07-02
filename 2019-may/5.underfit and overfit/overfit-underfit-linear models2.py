import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, linear_model, datasets
import numpy as np

#generate data with correlated features
X, y = datasets.make_classification(n_samples=100, n_features=100, n_informative=2, n_redundant=0, n_repeated=95, n_classes=2, random_state=100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

np.corrcoef(X_train, rowvar=False)

#overfit
perceptron_estimator = linear_model.Perceptron(max_iter=1000)
perceptron_grid  = {'alpha':[0] }
final_estimator = cutils.grid_search_best_model(perceptron_estimator, perceptron_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)

#overfit control
perceptron_estimator = linear_model.Perceptron(max_iter=1000)
perceptron_grid  = {'penalty':['l1', 'l2'], 'alpha':[0, 0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 3] }
final_estimator = cutils.grid_search_best_model(perceptron_estimator, perceptron_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)

#overfitting
lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'C':[1e50] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)

#overfit control
lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.001, 0.1, 0.5, 1, 10, 1e2] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)

