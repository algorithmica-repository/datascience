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
            ('feature_selector', feature_selection.SelectKBest(score_func=feature_selection.f_classif)),
            ('classifier', ensemble.GradientBoostingClassifier())
        ]

pipeline = pipeline.Pipeline(stages)
pipeline_grid  = {'feature_selector__k':[70, 75], 'classifier__max_depth':[1,2,3], 'classifier__n_estimators':list(range(50,300, 100)), 'classifier__learning_rate':[0.001, 0.1, 1.0]}
pipeline_generated = cutils.grid_search_best_model(pipeline, pipeline_grid, X_train, y_train, scoring="roc_auc")
final_estimator = pipeline_generated.named_steps['classifier']
print(pipeline_generated.score(X_eval, y_eval))

test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(test.info())
print(test.columns)

test1 = test.iloc[:,1:] 
test['target'] = pipeline_generated.predict_proba(test1)[:,1]
test.to_csv(os.path.join(dir, 'submission.csv'), columns=['id', 'target'], index=False)