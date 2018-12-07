import os
import pandas as pd
from sklearn import preprocessing, tree, neighbors, metrics, linear_model
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np
from sklearn import model_selection, ensemble
import seaborn as sns
from sklearn import feature_selection
import math

def get_continuous_features(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_features(df):
    return df.select_dtypes(exclude=['number']).columns

def cast_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def impute_categorical_features(df, features):    
    feature_defs = []
    for col_name in features:
        feature_defs.append((col_name, CategoricalImputer()))
    mapper = DataFrameMapper(feature_defs, input_df=True, df_out=True)
    df[features] = mapper.fit_transform(df[features])

def impute_continuous_features(df, features):
    cont_imputer = preprocessing.Imputer()
    cont_imputer.fit(df[features])
    print(cont_imputer.statistics_)
    df[features] = cont_imputer.transform(df[features])

def get_features_to_drop_on_missingdata(df, threshold) :
    tmp = df.isnull().sum()
    return list(tmp[tmp/float(df.shape[0]) > threshold].index)

def drop_features(df, features):
    return df.drop(features, axis=1, inplace=False)

def transform_cat_cont(df, features):
    return pd.get_dummies(df, columns = features)

def get_scaled_data(df):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(df)

def select_features_from_model(selector, X_train, y_train):
    selector.fit(X_train, y_train)
    plot_feature_importances(selector, X_train, y_train)
    return feature_selection.SelectFromModel(selector, prefit=True)

def plot_feature_importances(estimator, X_train, y_train):
    indices = np.argsort(estimator.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = estimator.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("RF feature importances")   

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )
 
def fit_model_objective(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse), cv=10)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.cv_results_)
   print(grid_estimator.best_params_)
   final_model = grid_estimator.best_estimator_
   print(final_model.coef_)
   print(final_model.intercept_)
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return final_model

def fit_model_neighbors(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse), cv=10)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.cv_results_)
   print(grid_estimator.best_params_)
   final_model = grid_estimator.best_estimator_
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return final_model

def fit_model_tree(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse), cv=10)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.cv_results_)
   print(grid_estimator.best_params_)
   final_model = grid_estimator.best_estimator_
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return final_model

def fit_model_ensemble(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse), cv=10)
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.cv_results_)
   print(grid_estimator.best_params_)
   final_model = grid_estimator.best_estimator_
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return final_model

path = 'D:/'
house_train = pd.read_csv(os.path.join(path,"house-train.csv"))
house_train.shape
house_train.info()

house_test = pd.read_csv(os.path.join(path,"house-test.csv"))
house_test.shape
house_test.info()

#combine train and test data to deal with level mismatch problem
house = pd.concat((house_train, house_test), axis = 0)
house.shape
house.info()

#type cast features
features_to_cast = ['MSSubClass']
cast_cont_to_cat(house, features_to_cast)

#retrieve continuous & categorical features
get_continuous_features(house)
get_categorical_features(house)

#drop non-useful features
features_to_drop = ['Id', 'SalePrice']
features_to_drop.extend(get_features_to_drop_on_missingdata(house, 0.25))
house1 = drop_features(house, features_to_drop)
house1.info()

impute_categorical_features(house1, get_categorical_features(house1))
impute_continuous_features(house1, get_continuous_features(house1))
house1.info()

house1 = transform_cat_cont(house1, get_categorical_features(house1))
print(house1.info())

y_train = house_train['SalePrice']
X_train = house1[:house_train.shape[0]]

#random forest regressor
rf_selector = ensemble.RandomForestRegressor(random_state=100)
selector = select_features_from_model(rf_selector, X_train, y_train)
X_train1 = selector.transform(X_train)

X_train1 = get_scaled_data(X_train1)

#knn regressor
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':list(range(2,20,1)), 'weights':['uniform', 'distance'] }
final_estimator = fit_model_neighbors(knn_estimator, knn_grid, X_train1, y_train)

X_test = house1[house_train.shape[0]:]
X_test = selector.transform(X_test)
X_test1 = get_scaled_data(X_test)

house_test['SalePrice'] = final_estimator.predict(X_test1)
house_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["Id", "SalePrice"], index=False)

