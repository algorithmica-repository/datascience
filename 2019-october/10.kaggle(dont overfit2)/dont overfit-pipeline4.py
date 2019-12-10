import pandas as pd
import os
from sklearn import tree, ensemble, pipeline, linear_model, metrics, model_selection, preprocessing, decomposition, manifold, feature_selection, svm
import seaborn as sns
import numpy as np

import sys
sys.path.append("G:/New Folder/utils")

import classification_utils as cutils
import common_utils as utils

dir = 'G:/10.kaggle(dont-overfit2)'
train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(train.info())
print(train.columns)

sns.countplot(x='target',data=train)

#filter unique value features
train1 = train.iloc[:,2:] 
y = train['target'].astype(int)

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(train1, y, test_size=0.1, random_state=1)

stages = [  ('imputer', preprocessing.Imputer()),
            ('zv_filter', feature_selection.VarianceThreshold()),
            ('feature_selector', feature_selection.RFE(svm.LinearSVC(max_iter=10000)) ),
            ('classifier', linear_model.LogisticRegression())
        ]
pipeline = pipeline.Pipeline(stages)
pipeline_grid  = {'feature_selector__n_features_to_select':[10, 20], 'classifier__C':[0.001, 0.01, 0.1, 0.2, 0.5],'classifier__penalty':['l1', 'l2'], 'classifier__class_weight':['balanced', None]}
pipeline_generated = cutils.grid_search_best_model(pipeline, pipeline_grid, X_train, y_train, scoring="roc_auc")
final_estimator = pipeline_generated.named_steps['classifier']
print(pipeline_generated.score(X_eval, y_eval))

test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(test.info())
print(test.columns)

test1 = test.iloc[:,1:] 
test['target'] = pipeline_generated.predict_proba(test1)[:,1]
test.to_csv(os.path.join(dir, 'submission.csv'), columns=['id', 'target'], index=False)