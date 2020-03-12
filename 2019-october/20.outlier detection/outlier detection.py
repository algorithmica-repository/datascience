import sys
path = 'F://New Folder/utils'
sys.path.append(path)

import common_utils  as utils
import regression_utils as rutils
from sklearn import metrics, model_selection, ensemble, neighbors, linear_model, decomposition, manifold, feature_selection, preprocessing, pipeline, impute, compose, svm, cluster
import math
import pandas as pd
import os
import numpy as np

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
house_train2 = preprocess_pipeline.fit_transform(house_train1)

#outlier detection pipeline     
outlier_pipeline = pipeline.Pipeline([
                     ('preprocess', preprocess_pipeline),
                     ('outlier_estimator', ensemble.IsolationForest(contamination=0.01))
                ])
labels = outlier_pipeline.fit_predict(house_train1)
house_train1[labels==-1]
    
#add clustering label as new featuer
clustering = cluster.AgglomerativeClustering(n_clusters=5)
clustering.fit(house_train2)
house_train2['house_group'] = clustering.labels_
  
#feature reduction by clustering related features      
cluster_pipeline = pipeline.Pipeline([
                     ('preprocess', preprocess_pipeline),
                     ('cluster_features', cluster.FeatureAgglomeration(n_clusters=50))
                ])
cluster_data = cluster_pipeline.fit_transform(house_train1)
    
#feature reduction on top of correlation matrix
corr_matrix = np.corrcoef(house_train2, rowvar=False)
clustering = cluster.AgglomerativeClustering(n_clusters=10)
clustering.fit(corr_matrix)

