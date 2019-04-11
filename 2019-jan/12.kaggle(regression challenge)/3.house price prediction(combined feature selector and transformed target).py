import sys
path_to_scripts = 'G://'
sys.path.append(path_to_scripts)

from common_utils  import get_continuous_features, get_categorical_features, \
    cast_cont_to_cat, get_categorical_imputers, get_continuous_imputers, \
    get_features_to_drop_on_missingdata, ohe, drop_features, get_scaler, \
    get_zero_variance_filter, feature_reduction_pca, feature_reduction_tsne, \
    plot_feature_importances, select_features, grid_search_plot_one_parameter_curves, \
    grid_search_plot_two_parameter_curves, get_best_model
from regression_utils import generate_nonlinear_synthetic_data_regression, generate_linear_synthetic_data_regression, \
    plot_model_2d_regression, plot_model_3d_regression, plot_data_2d_regression, plot_data_3d_regression, \
    grid_search_plot_models_regression, plot_coefficients_regression, \
    plot_target_and_transformed_target_regression, rmse
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree, linear_model, ensemble, neighbors, svm, feature_selection
import sklearn
import pandas as pd
import os
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt

def log_rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )

path = 'G://'
house_train = pd.read_csv(os.path.join(path,"train.csv"))
house_train.shape
house_train.info()

house_test = pd.read_csv(os.path.join(path,"test.csv"))
house_test.shape
house_test.info()

house = pd.concat((house_train, house_test), axis = 0)
house.shape
house.info()

features_to_cast = ['MSSubClass']
cast_cont_to_cat(house, features_to_cast)

print(get_continuous_features(house))
print(get_categorical_features(house))

features_to_drop = ['Id', 'SalePrice']
features_to_drop.extend(get_features_to_drop_on_missingdata(house, 0.25))
house1 = drop_features(house, features_to_drop)
house1.info()

imputable_cat_features = get_categorical_features(house1)
cat_imputer = get_categorical_imputers(house1, imputable_cat_features)
house1[imputable_cat_features] = cat_imputer.transform(house1[imputable_cat_features])

imputable_cont_features = get_continuous_features(house1)
cont_imputer = get_continuous_imputers(house1, imputable_cont_features)
house1[imputable_cont_features] = cont_imputer.transform(house1[imputable_cont_features])
house1.info()

house2 = ohe(house1, imputable_cat_features)

scaler = get_scaler(house2)
house3 = scaler.transform(house2)
house3 = pd.DataFrame(house3, columns=house2.columns)

X_train = house3[:house_train.shape[0]]
y_train = house_train['SalePrice']
sns.distplot(y_train, hist=True)
y_trans = np.log1p(y_train)
sns.distplot(y_trans, hist=True)

#union of 3 feature selectors
lasso_estimator = linear_model.Lasso()
lasso_grid = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]}
lasso_selector = get_best_model(lasso_estimator, lasso_grid, X_train, y_trans, scoring = scoring)
plot_feature_importances1(lasso_selector, X_train, 50)
important_features_lasso = get_important_features(lasso_selector, X_train)

rf_estimator = ensemble.RandomForestRegressor(random_state=100)
rf_grid = {'n_estimators':list(range(100,501,200)), 'max_features':[14, 16, 18, 20], 'max_depth':[3,5,7]}
rf_selector = get_best_model(rf_estimator, rf_grid, X_train, y_trans, scoring=scoring)
plot_feature_importances1(rf_selector, X_train, 50)
important_features_rf = get_important_features(rf_selector, X_train)

dt_estimator = tree.DecisionTreeRegressor()
ada_estimator = ensemble.AdaBoostRegressor(dt_estimator)
ada_grid = {'n_estimators':list(range(100,501,200)), 'learning_rate':[0.1,1.0], 'base_estimator__max_depth':[1,3,5]}
ada_selector = get_best_model(ada_estimator, ada_grid, X_train, y_trans, scoring=scoring)
plot_feature_importances1(ada_selector, X_train, 50)
important_features_ada = get_important_features(ada_selector, X_train)

important_features = set(important_features_lasso) | set(important_features_rf) | set(important_features_ada)
print(len(important_features))

X_train1 = X_train[list(important_features)]

pca_data = feature_reduction_pca(X_train1, X_train1.shape[1])
tsne_data = feature_reduction_tsne(X_train1, 2)
plot_data_3d_regression(tsne_data, y_train)

scoring = metrics.make_scorer(log_rmse, greater_is_better=False)

gb_estimator = ensemble.GradientBoostingRegressor()
gb_grid = {'n_estimators':list(range(100,501,200)), 'learning_rate':[0.1,1.0], 'max_depth':[1,3,5]}
gb_model = get_best_model(gb_estimator, gb_grid, X_train1, y_trans, scoring=scoring)

svm_estimator = svm.SVR(kernel='rbf')
svm_grid= {'C':[1, 15, 30], 'gamma':[0.1, 0.2] }
svm_model = get_best_model(svm_estimator, svm_grid, X_train1, y_trans, scoring = scoring )

X_test = house3[house_train.shape[0]:]
X_test1 = X_test[list(important_features)]

house_test['SalePrice'] = np.expm1(gb_model.predict(X_test1))
path='C:/Users/Thimma Reddy'
house_test.to_csv(os.path.join(path, "submission.csv"), columns=["Id", "SalePrice"], index=False)
