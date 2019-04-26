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

dt_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=dt_estimator)
ada_grid = {'n_estimators':[100], 'base_estimator__max_depth':list(range(1,4)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_classification(ada_estimator, ada_grid, X_train, y_train)
grid_search_plot_two_parameter_curves(ada_estimator, ada_grid, X_train, y_train, scoring =  scoring)
ada_final_model = grid_search_best_model(ada_estimator, ada_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(ada_final_model, X_train, y_train)
performance_metrics_hard_binary_classification(ada_final_model, X_test, y_test)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,5)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_classification(gb_estimator, gb_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(gb_estimator, gb_grid, X_train, y_train, scoring =  scoring)
gb_final_model = grid_search_best_model(gb_estimator, gb_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(gb_final_model, X_train, y_train)
performance_metrics_hard_binary_classification(ada_final_model, X_test, y_test)

#non-linear multi-class classification
X, y = generate_nonlinear_synthetic_data_classification1(1000, 2, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
plot_data_2d_classification(X_train, y_train)

dt_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=dt_estimator)
ada_grid = {'n_estimators':[100], 'base_estimator__max_depth':list(range(1,4)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_classification(ada_estimator, ada_grid, X_train, y_train)
grid_search_plot_two_parameter_curves(ada_estimator, ada_grid, X_train, y_train, scoring =  scoring)
ada_final_model = grid_search_best_model(ada_estimator, ada_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(ada_final_model, X_train, y_train)
performance_metrics_hard_multiclass_classification(ada_final_model, X_test, y_test)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid = {'n_estimators':list(range(10,100,40)), 'max_depth':list(range(3,5)), 'learning_rate':[0.1,0.5]}
grid_search_plot_models_classification(gb_estimator, gb_grid, X_train, y_train )
grid_search_plot_two_parameter_curves(gb_estimator, gb_grid, X_train, y_train, scoring =  scoring)
gb_final_model = grid_search_best_model(gb_estimator, gb_grid, X_train, y_train, scoring = scoring )
plot_model_2d_classification(gb_final_model, X_train, y_train)
performance_metrics_hard_multiclass_classification(ada_final_model, X_test, y_test)