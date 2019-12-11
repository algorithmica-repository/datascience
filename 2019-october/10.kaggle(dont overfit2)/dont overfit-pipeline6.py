import pandas as pd
import os
from sklearn import tree, ensemble, pipeline, naive_bayes, linear_model, metrics, model_selection, preprocessing, decomposition, manifold, feature_selection, svm
import seaborn as sns
import sys
sys.path.append("E:/New Folder/utils")
import tpot

dir = 'E:/10.kaggle(dont-overfit2)'
train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(train.info())
print(train.columns)

sns.countplot(x='target',data=train)

#filter unique value features
train1 = train.iloc[:,2:] 
y = train['target'].astype(int)

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(train1, y, test_size=0.1, random_state=1)

tpot_estimator = tpot.TPOTClassifier(generations=10, population_size=40, 
                                     verbosity=2, early_stop=2, 
                                     random_state=100,
                                     cv=5, scoring='roc_auc',
                                     config_dict=None, warm_start=True,
                                     periodic_checkpoint_folder='E:/checkpoint')

tpot_estimator.fit(X_train, y_train)
print(tpot_estimator.score(X_train, y_train))
print(tpot_estimator.evaluated_individuals_)
print(tpot_estimator.fitted_pipeline_)

print(tpot_estimator.score(X_eval, y_eval))

test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(test.info())
print(test.columns)

test1 = test.iloc[:,1:] 
test['target'] = tpot_estimator.predict_proba(test1)[:,1]
test.to_csv(os.path.join(dir, 'submission.csv'), columns=['id', 'target'], index=False)
