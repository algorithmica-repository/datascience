import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
from sklearn import model_selection, linear_model, dummy

#binary classification
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.4,0.6], class_sep=1.5)
#X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)


X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.01, 0.001, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring='accuracy')
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)
cutils.performance_metrics_hard_binary_classification(final_estimator, X_eval, y_eval)

#multi class classification
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=4, weights=[0.3,0.3,0.2,0.2], class_sep=1.5)
#X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)


X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()
lr_grid  = {'penalty':['l1', 'l2'], 'C':[0.01, 0.001, 0.1, 0.3, 0.5, 0.7, 1] }
final_estimator = cutils.grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring='accuracy')
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)
cutils.performance_metrics_hard_multiclass_classification(final_estimator, X_eval, y_eval)
