import sys
path_to_scripts = 'E://'
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
from sklearn import tree, linear_model, ensemble, neighbors
import sklearn
import pandas as pd
import os
import numpy as np
import seaborn as sns
import math

def log_rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )


path = 'E://'
house_train = pd.read_csv(os.path.join(path,"train.csv"))
house_train.shape
house_train.info()

house_test = pd.read_csv(os.path.join(path,"test.csv"))
house_test.shape
house_test.info()

house = pd.concat((house_train, house_test), axis = 0)
house.shape
house.info()

print(get_continuous_features(house))
print(get_categorical_features(house))

features_to_cast = ['MSSubClass']
cast_cont_to_cat(house, features_to_cast)

print(get_continuous_features(house))
print(get_categorical_features(house))

features_to_drop = ['Id']
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

lasso_selector = linear_model.Lasso()
lasso_selector.fit(X_train, y_train)
print(lasso_selector.coef_)
plot_feature_importances(lasso_selector, X_train, 40)

X_train1 = select_features(lasso_selector, X_train)

#corr_heatmap(X_train1)
pca_data = feature_reduction_pca(X_train1, X_train1.shape[1])
tsne_data = feature_reduction_tsne(X_train1, 2)
plot_data_3d_regression(tsne_data, y_train)

scoring = metrics.make_scorer(log_rmse, greater_is_better=False)

knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':list(range(5,15)) }
grid_search_plot_one_parameter_curves(knn_estimator, knn_grid, X_train1, y_train, scoring = scoring)
final_model = get_best_model(knn_estimator, knn_grid, X_train1, y_train, scoring=scoring)

X_test = house3[house_train.shape[0]:]
X_test1 = select_features(lasso_selector, X_test)

house_test['SalePrice'] = final_model.predict(X_test1)
house_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["Id", "SalePrice"], index=False)
