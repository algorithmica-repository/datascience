import sys
path = 'F://New Folder/utils'
sys.path.append(path)

import common_utils  as utils
from sklearn import metrics, model_selection, ensemble, neighbors, linear_model, decomposition, manifold, feature_selection, preprocessing, pipeline, impute, compose, svm
import math
import pandas as pd
import os
import numpy as np
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline


def log_rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )

path = 'F://house-prices'
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

    #build feature selection pipeline
features_pipeline = pipeline.FeatureUnion([
                    ('pca_selector', decomposition.PCA() ),
                    ('et_selector', feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier()) )
                ])



regressor = svm.LinearSVR()
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
                  'tregressor__regressor__C':[0.01, 0.1, 5, 10]
                  }

#build model with pipeline
scoring = metrics.make_scorer(log_rmse, greater_is_better=False)
pipeline_generated = utils.grid_search_best_model(complete_pipeline, pipeline_grid, house_train1, house_train['SalePrice'], scoring=scoring)
print(pipeline_generated)

objects_to_dump = {
        'features_to_cast': features_to_cast,
        'features_to_drop': features_to_drop,
        'pipeline': pipeline_generated
        }
joblib.dump(objects_to_dump, os.path.join(path, 'house_price_model_v1.pkl'))

#build pipeline in pmml format
complete_pipeline_pmml = PMMLPipeline([  
                    ('preprocess', preprocess_pipeline),
                    ('zv_filter', feature_selection.VarianceThreshold() ),
                    ('features', features_pipeline ),
                    ('tregressor', compose.TransformedTargetRegressor(
                                        regressor= regressor,
                                        func=np.log1p, inverse_func=np.expm1))
                ])

pipeline_grid  = {'features__pca_selector__n_components':[2, 3],
                  'tregressor__regressor__C':[0.01, 0.1, 5, 10]
                  }

pipeline_generated_pmml = utils.grid_search_best_model(complete_pipeline_pmml, pipeline_grid, house_train1, house_train['SalePrice'], scoring=scoring)
sklearn2pmml(pipeline_generated_pmml, 'house_price_model_v1.pmml', with_repr = True)
