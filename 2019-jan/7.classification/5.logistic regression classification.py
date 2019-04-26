import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'E://'
sys.path.append(path)

from common_utils  import *
from classification_utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection

#linear binary classification
X, y = generate_linear_synthetic_data_classification(2000, 2, 2, [.5,.5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_classification(X_train, y_train)

scoring = 'accuracy'

lr_estimator = linear_model.LogisticRegression()                     
lr_grid = {'C':[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0] }
grid_search_plot_models_classification(lr_estimator, lr_grid, X_train, y_train)
grid_search_plot_one_parameter_curves(lr_estimator, lr_grid, X_train, y_train, scoring=scoring)
lr_final_model = grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring=scoring)
plot_model_2d_classification(lr_final_model, X_train, y_train)
y_prob = lr_final_model.predict_proba(X_test)
performance_metrics_hard_binary_classification(lr_final_model, X_test, y_test)

#linear multi-class classification
X, y = generate_linear_synthetic_data_classification(2000, 2, 3, [.3,.3,.4])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_classification(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()                     
lr_grid = {'C':[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0] }
grid_search_plot_models_classification(lr_estimator, lr_grid, X_train, y_train)
grid_search_plot_one_parameter_curves(lr_estimator, lr_grid, X_train, y_train, scoring=scoring)
lr_final_model = grid_search_best_model(lr_estimator, lr_grid, X_train, y_train, scoring=scoring)
plot_model_2d_classification(lr_final_model, X_train, y_train)
performance_metrics_hard_multiclass_classification(lr_final_model, X_test, y_test)
