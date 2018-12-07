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
    select = feature_selection.SelectFromModel(selector, prefit=True)
    return select.transform(X_train)

def plot_feature_importances(estimator, X_train, y_train):
    indices = np.argsort(estimator.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = estimator.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("RF feature importances")   

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )
 
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

#changes working directory
path = 'D:/'
house_train = pd.read_csv(os.path.join(path,"house-train.csv"))
house_train.shape
house_train.info()

#type cast features
features_to_cast = ['MSSubClass']
cast_cont_to_cat(house_train, features_to_cast)

#retrieve continuous & categorical features
get_continuous_features(house_train)
get_categorical_features(house_train)

#drop non-useful features
features_to_drop = ['Id', 'SalePrice']
features_to_drop.extend(get_features_to_drop_on_missingdata(house_train, 0.25))
house_train1 = drop_features(house_train, features_to_drop)
house_train1.info()

impute_categorical_features(house_train1, get_categorical_features(house_train1))
impute_continuous_features(house_train1, get_continuous_features(house_train1))
house_train1.info()

house_train1 = transform_cat_cont(house_train1, get_categorical_features(house_train1))
print(house_train1.info())

y_train = house_train['SalePrice']

#random forest regressor
rf_selector = ensemble.RandomForestRegressor(random_state=100)
X_train = select_features_from_model(rf_selector, house_train1, y_train)

X_train = get_scaled_data(X_train)

#tree regressor
dt_estimator = tree.DecisionTreeRegressor(random_state=100)
dt_grid = {'max_depth':list(range(3,10,1)), 'criterion':['entropy', 'gini'], 'max_features':[5,10,20,30] }
fit_model_tree(dt_estimator, dt_grid, X_train, y_train)

#knn regressor
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':list(range(2,20,1)), 'weights':['uniform', 'distance'] }
fit_model_neighbors(knn_estimator, knn_grid, X_train, y_train)

#gradient boosting regressor
gb_estimator = ensemble.GradientBoostingRegressor()
gb_grid = {'n_estimators':list(range(50,501,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5, 7]}
fit_model_ensemble(gb_estimator, gb_grid, X_train, y_train)

#linear regressor & its variations
linr_estimator = linear_model.LinearRegression()
linr_grid = {'fit_intercept':['False','True']}
fit_model_objective(linr_estimator, linr_grid, X_train, y_train)

ridge_estimator = linear_model.Ridge()
ridge_grid = {'alpha':[0.1,0.5,1.0]}
fit_model_objective(ridge_estimator, ridge_grid, X_train, y_train)

lasso_estimator = linear_model.Lasso()
lasso_grid = {'alpha':[0.1,0.5,1.0]}
fit_model_objective(lasso_estimator, lasso_grid, X_train, y_train)

enet_estimator = linear_model.ElasticNet()
enet_grid = {'alpha':[0.1,0.5], 'l1_ratio':[0.1,0.2,0.5]}
fit_model_objective(enet_estimator, enet_grid, X_train, y_train)


