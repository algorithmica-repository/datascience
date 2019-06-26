import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, linear_model, svm

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.5, 0.5], class_sep=3)
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

#perceptron algorithm
perceptron_estimator = linear_model.Perceptron()
perceptron_grid  = {'penalty':['l1', 'l2'], 'alpha':[0, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(perceptron_estimator, perceptron_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

#predict distances and classes for test data
print(final_estimator.decision_function(X_test))
print(final_estimator.predict(X_test))

#logistic regression algorithm
lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.01, 0.001, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

#predict distances and classes for test data
print(final_estimator.decision_function(X_test))
print(final_estimator.predict(X_test))
print(final_estimator.predict_proba(X_test))

#linear svm algorithm
lsvm_estimator = svm.LinearSVC()
lsvm_grid  = {'penalty':['l2'], 'C':[0.1, 0.3, 0.5, 0.7, 1, 10] }
final_estimator = cutils.grid_search_best_model(lsvm_estimator, lsvm_grid, X_train, y_train)
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

#predict distances and classes for test data
print(final_estimator.decision_function(X_test))
print(final_estimator.predict(X_test))
