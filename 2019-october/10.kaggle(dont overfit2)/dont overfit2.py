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

rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(1,9)), 'n_estimators':list(range(1,300,100)) }
rf_final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, train1, y)
embedded_selector = feature_selection.SelectFromModel(rf_final_estimator, prefit=True, threshold='mean')
train3 = embedded_selector.transform(train1)
utils.plot_feature_importances(rf_final_estimator,train1, cutoff=50)

et_estimator = ensemble.ExtraTreesClassifier()
et_grid  = {'max_depth':list(range(1,9)), 'n_estimators':list(range(1,300,100)) }
et_final_estimator = cutils.grid_search_best_model(et_estimator, et_grid, train1, y)
embedded_selector = feature_selection.SelectFromModel(et_final_estimator, prefit=True, threshold='mean')
train3 = embedded_selector.transform(train1)
utils.plot_feature_importances(et_final_estimator,train1, cutoff=50)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,300, 100)), 'learning_rate':[0.001, 0.1, 1.0] }
gb_final_estimator = cutils.grid_search_best_model(gb_estimator, gb_grid, train1, y)
embedded_selector = feature_selection.SelectFromModel(gb_final_estimator, prefit=True, threshold='mean')
X_train1 = embedded_selector.transform(train1)
utils.plot_feature_importances(gb_final_estimator, train1)

kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.01, 0.1, 1, 2, 5, 10], 'C':[0.001, 0.01, 0.1, 0.5] }
final_estimator = cutils.grid_search_best_model(kernel_svm_estimator, kernel_svm_grid, train1, y)
embedded_selector = feature_selection.SelectFromModel(final_estimator, prefit=True, threshold='mean')
X_train1 = embedded_selector.transform(train1)
utils.plot_feature_importances(final_estimator, train1)

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(train_pca, y, test_size=0.1, random_state=1)

sns.countplot(x='target',data=train)

kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.01, 0.1, 1, 2, 5, 10], 'C':[0.001, 0.01, 0.1, 0.5] }
final_estimator = cutils.grid_search_best_model(kernel_svm_estimator, kernel_svm_grid, X_train, y_train)

print(final_estimator.score(X_eval, y_eval))

test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(test.info())
print(test.columns)

test1 = test.iloc[:,1:] 
test2 = variance.transform(test1)
test_pca = lpca.transform(test2)
test['target'] = final_estimator.predict(test_pca)
test.to_csv(os.path.join(dir, 'submission.csv'), columns=['id', 'target'], index=False)
