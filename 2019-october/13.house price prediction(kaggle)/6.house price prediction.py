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

#build feature selection pipeline
features_pipeline = pipeline.FeatureUnion([
                    ('pca_selector', decomposition.PCA() ),
                    ('et_selector', feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier()) )
                ])



regressor = svm.SVR(kernel="rbf")
#build complete pipeline with feature selection and ml algorithms
complete_pipeline = pipeline.Pipeline([  
                    ('preprocess', preprocess_pipeline),
                    ('zv_filter', feature_selection.VarianceThreshold() ),
                    ('features', features_pipeline ),
                    ('tregressor', compose.TransformedTargetRegressor(
                                        regressor= regressor,
                                        func=np.log1p, inverse_func=np.expm1))
                ])

pipeline_grid  = {'features__pca_selector__n_components':[2, 3],
                  'tregressor__regressor__gamma':[0.1, 0.5, 5, 10],
                  'tregressor__regressor__C':[0.01, 0.1, 5, 10]
                  }

#build model with pipeline
scoring = metrics.make_scorer(log_rmse, greater_is_better=False)
pipeline_generated = utils.grid_search_best_model(complete_pipeline, pipeline_grid, house_train1, house_train['SalePrice'], scoring=scoring)

#read test data
house_test = pd.read_csv(os.path.join(path,"test.csv"))
house_test.shape
house_test.info()
house_test['SalePrice'] = None

#apply preprocessing required before pipeline
utils.cast_to_cat(house_test, features_to_cast)
house_test1 = utils.drop_features(house_test, features_to_drop)
house_test1.info()

#get predictions on test data with constructed pipeline
house_test['SalePrice'] = np.round(pipeline_generated.predict(house_test1), 2)
house_test.to_csv(os.path.join(path, "submission.csv"), columns=["Id", "SalePrice"], index=False)
