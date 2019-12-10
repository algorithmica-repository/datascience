import pandas as pd
import os
from sklearn import tree, ensemble, model_selection, preprocessing, decomposition, manifold, feature_selection, svm
import seaborn as sns
import numpy as np

import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
import common_utils as utils

dir = 'C:/Users/Algorithmica/Downloads/dont-overfit-ii'
train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(train.info())
print(train.columns)

#filter unique value features
train1 = train.iloc[:,2:] 
y = train['target'].astype(int)

#filter zero-variance features
variance = feature_selection.VarianceThreshold()
train2 = variance.fit_transform(train1)

#embedded feature selection
rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(1,9)), 'n_estimators':list(range(1,300,100)) }
rf_final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, train1, y)
embedded_selector = feature_selection.SelectFromModel(rf_final_estimator, prefit=True, threshold='mean')
utils.plot_feature_importances(rf_final_estimator,train1, cutoff=50)
train2 = embedded_selector.transform(train1)

#statistical feature selection
statistical_selector = feature_selection.SelectKBest(feature_selection.f_classif, k = 20)
train2 = statistical_selector.fit_transform(train1, y)
print(statistical_selector.scores_)

#recursive feature elimination(rfe)
rf_estimator = ensemble.RandomForestClassifier()
rfe_selector = feature_selection.RFE(rf_estimator, n_features_to_select=10, step=5)
train2 = rfe_selector.fit_transform(train1, y)
print(rfe_selector.ranking_)
