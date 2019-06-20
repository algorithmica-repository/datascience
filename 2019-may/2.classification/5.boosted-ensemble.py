import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, ensemble, tree
import xgboost as xgb

X_train, y_train = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=500, noise=0.25)
cutils.plot_data_2d_classification(X_train, y_train)

dt_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(dt_estimator)
ada_grid  = {'base_estimator__max_depth':[1,2,3], 'n_estimators':list(range(50,150, 30)), 'learning_rate':[0.1, 0.2, 0.5, 1.0] }
final_estimator = cutils.grid_search_best_model(ada_estimator, ada_grid, X_train, y_train)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,150, 30)), 'learning_rate':[0.1, 0.2, 0.5, 1.0] }
final_estimator = cutils.grid_search_best_model(gb_estimator, gb_grid, X_train, y_train)

xgb_estimator = xgb.XGBClassifier()
xgb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,150, 30)), 'learning_rate':[0.1, 0.2, 0.5, 1.0], 'reg_alpha':[0, 0.5], 'reg_lambda':[0.5, 1] }
final_estimator = cutils.grid_search_best_model(xgb_estimator, xgb_grid, X_train, y_train)
