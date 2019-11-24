import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
from sklearn import model_selection, ensemble, tree, neighbors
import xgboost as xgb

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.5, 0.5], class_sep=2)
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

#grid search for parameter values
dt_estimator = tree.DecisionTreeClassifier()
dt_grid  = {'criterion':['gini', 'entropy'], 'max_depth':list(range(1,9)) }
final_estimator = cutils.grid_search_best_model(dt_estimator, dt_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid  = {'n_neighbors':list(range(1,21)), 'weights':['uniform', 'distance'] }
final_estimator = cutils.grid_search_best_model(knn_estimator, knn_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(5,10)), 'n_estimators':list(range(1,100, 20)) }
final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

et_estimator = ensemble.ExtraTreesClassifier()
et_grid  = {'max_depth':list(range(5,10)), 'n_estimators':list(range(1,100, 20)) }
final_estimator = cutils.grid_search_best_model(et_estimator, et_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

dt_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(dt_estimator)
ada_grid  = {'base_estimator__max_depth':[1,2,3], 'n_estimators':list(range(50,150, 30)), 'learning_rate':[0.1, 0.2, 0.5, 1.0] }
final_estimator = cutils.grid_search_best_model(ada_estimator, ada_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,150, 30)), 'learning_rate':[0.1, 0.2, 0.5, 1.0] }
final_estimator = cutils.grid_search_best_model(gb_estimator, gb_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

xgb_estimator = xgb.XGBClassifier()
xgb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,150, 30)), 'learning_rate':[0.1, 0.2, 0.5, 1.0], 'reg_alpha':[0, 0.5], 'reg_lambda':[0.5, 1] }
final_estimator = cutils.grid_search_best_model(xgb_estimator, xgb_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)
