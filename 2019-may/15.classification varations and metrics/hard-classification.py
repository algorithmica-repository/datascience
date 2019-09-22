import sys
sys.path.append("E:/utils")

import classification_utils as cutils
from sklearn import model_selection, linear_model, dummy

#balanced binary classification
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.4,0.6], class_sep=0.1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.01, 0.001, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring='accuracy')
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

cutils.performance_metrics_hard_binary_classification(final_estimator, X_test, y_test)

#imbalanced binary classification
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.05,0.95], class_sep=0.1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.01, 0.001, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring='f1')
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

cutils.performance_metrics_hard_binary_classification(final_estimator, X_test, y_test)

dummy_estimator = dummy.DummyClassifier(strategy='most_frequent')
dummy_estimator.fit(X_train, y_train)
dummy_estimator.predict(X_test)
cutils.performance_metrics_hard_binary_classification(dummy_estimator, X_test, y_test)


#imbalanced multi-class classification
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=3, weights=[0.6,0.3,0.1], class_sep=1.5)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.01, 0.001, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring='f1_macro')
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

cutils.performance_metrics_hard_multiclass_classification(final_estimator, X_test, y_test)

dummy_estimator = dummy.DummyClassifier(strategy='most_frequent')
dummy_estimator.fit(X_train, y_train)
dummy_estimator.predict(X_test)
cutils.performance_metrics_hard_multiclass_classification(dummy_estimator, X_test, y_test)
