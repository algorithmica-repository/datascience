import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, ensemble, tree

X_train, y_train = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=500, noise=0.25)
cutils.plot_data_2d_classification(X_train, y_train)

dt_estimator = tree.DecisionTreeClassifier()
bag_estimator = ensemble.BaggingClassifier(dt_estimator)
bag_grid  = {'base_estimator__max_depth':list(range(5,8)), 'n_estimators':list(range(1,100, 20)) }
final_estimator = cutils.grid_search_best_model(bag_estimator, bag_grid, X_train, y_train)

rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(5,10)), 'n_estimators':list(range(1,100, 20)) }
final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, X_train, y_train)

et_estimator = ensemble.ExtraTreesClassifier()
et_grid  = {'max_depth':list(range(5,10)), 'n_estimators':list(range(1,100, 20)) }
final_estimator = cutils.grid_search_best_model(et_estimator, et_grid, X_train, y_train)
