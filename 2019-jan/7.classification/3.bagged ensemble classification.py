import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

from common_utils  import *
from classification_utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection

#non-linear binary classification
X, y = generate_nonlinear_synthetic_data_classification1(1000, 2, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_classification(X_train, y_train)

scoring = 'accuracy'

rf_estimator = ensemble.RandomForestClassifier()
rf_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_models_classification(rf_estimator, rf_grid, X_train, y_train)
grid_search_plot_two_parameter_curves(rf_estimator, rf_grid, X_train, y_train, scoring =  scoring)
rf_final_model = grid_search_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(rf_final_model, X_train, y_train)
performance_metrics_hard_binary_classification(rf_final_model, X_test, y_test)

et_estimator = ensemble.ExtraTreesClassifier()
et_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(et_estimator, et_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_classification(et_estimator, et_grid, X_train, y_train )
et_final_model = get_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(et_final_model, X_train, y_train)
performance_metrics_hard_binary_classification(et_rf_final_model, X_test, y_test)


#non-linear multi-class classification
X, y = generate_nonlinear_synthetic_data_classification1(1000, 2, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_classification(X_train, y_train)

rf_estimator = ensemble.RandomForestClassifier()
rf_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_models_classification(rf_estimator, rf_grid, X_train, y_train)
grid_search_plot_two_parameter_curves(rf_estimator, rf_grid, X_train, y_train, scoring =  scoring)
rf_final_model = grid_search_best_model(rf_estimator, rf_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(rf_final_model, X_train, y_train)
performance_metrics_hard_multiclass_classification(rf_final_model, X_test, y_test)

et_estimator = ensemble.ExtraTreesClassifier()
et_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,6))}
grid_search_plot_two_parameter_curves(et_estimator, et_grid, X_train, y_train, scoring =  scoring)
grid_search_plot_models_classification(et_estimator, et_grid, X_train, y_train )
et_final_model = get_best_model(et_estimator, et_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(et_final_model, X_train, y_train)
performance_metrics_hard_multiclass_classification(et_rf_final_model, X_test, y_test)