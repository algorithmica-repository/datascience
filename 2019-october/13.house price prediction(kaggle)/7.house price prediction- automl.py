import sys
path = 'I://New Folder/utils'
sys.path.append(path)

import common_utils  as utils
import regression_utils as rutils
from sklearn import metrics, model_selection, ensemble, neighbors, linear_model, decomposition, manifold, feature_selection, preprocessing, pipeline, impute, compose, svm
import math
import pandas as pd
import os
import numpy as np
import tpot

def log_rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )

path = 'I://house-prices'
house_train = pd.read_csv(os.path.join(path,"train.csv"))
house_train.shape
house_train.info()

#type cast features
features_to_cast = ['MSSubClass']
utils.cast_to_cat(house_train, features_to_cast)

#manual feature selection
features_to_drop = ['Id', 'SalePrice']
missing_features_above_th = utils.get_features_to_drop_on_missingdata(house_train, 0.25)
features_to_drop.extend(missing_features_above_th)
house_train1 = utils.drop_features(house_train, features_to_drop)
house_train1.shape

#build pipeline for categorical features
categorical_pipeline = pipeline.Pipeline([ 
                    ('imputer', impute.SimpleImputer(strategy="most_frequent") ),
                    ('ohe', preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore') )
               ])


#build pipeline for numerical features
numerical_pipeline = pipeline.Pipeline([
                    ('imputer', impute.SimpleImputer() ),
                    ('scaler', preprocessing.StandardScaler() )
                ])

#build preprocessing pipeline for all features
cat_features = utils.get_non_continuous_features(house_train1)
num_features = utils.get_continuous_features(house_train1)

preprocess_pipeline = compose.ColumnTransformer([
                    ('cat', categorical_pipeline, cat_features),
                    ('num', numerical_pipeline, num_features)
                ])

viz_pipeline = pipeline.Pipeline([
                     ('preprocess', preprocess_pipeline),
                     ('pca', decomposition.PCA(n_components=0.95)),
                     ('tsne', manifold.TSNE(2))
                ])

tsne_data = viz_pipeline.fit_transform(house_train1)
rutils.plot_data_3d_regression(tsne_data, house_train['SalePrice'])

X_train = preprocess_pipeline.fit_transform(house_train1)
y_train = house_train['SalePrice']

scoring = metrics.make_scorer(log_rmse, greater_is_better=False)
tpot_estimator = tpot.TPOTRegressor(generations=10, population_size=40, 
                                     verbosity=2, early_stop=2, 
                                     random_state=100,
                                     cv=5, scoring=scoring, 
                                     config_dict=None, warm_start=True,
                                     periodic_checkpoint_folder='I:/checkpoint')
tpot_estimator.fit(X_train, y_train)
print(tpot_estimator.score(X_train, y_train))
print(tpot_estimator.evaluated_individuals_)
print(tpot_estimator.fitted_pipeline_)

#read test data
house_test = pd.read_csv(os.path.join(path,"test.csv"))
house_test.shape
house_test.info()
house_test['SalePrice'] = None

#apply preprocessing required before pipeline
utils.cast_to_cat(house_test, features_to_cast)
house_test1 = utils.drop_features(house_test, features_to_drop)
house_test1.info()
X_test = preprocess_pipeline.fit_transform(house_test1)


#get predictions on test data with constructed pipeline
house_test['SalePrice'] = np.round(tpot_estimator.predict(house_test1), 2)
house_test.to_csv(os.path.join(path, "submission.csv"), columns=["Id", "SalePrice"], index=False)
