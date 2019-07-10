import sys
sys.path.append("G:/")

import pandas as pd
import os
import common_utils as utils
from sklearn import preprocessing, neighbors, svm, linear_model, ensemble, pipeline, model_selection
import classification_utils as cutils
import kernel_utils as kutils
import tpot

dir = 'G:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))

print(titanic_train.shape)
print(titanic_train.info())

titanic_train1 = utils.drop_features(titanic_train, ['PassengerId', 'Name', 'Survived', 'Ticket', 'Cabin'])

#type casting
utils.cast_to_cat(titanic_train1, ['Sex', 'Pclass', 'Embarked'])

cat_features = utils.get_categorical_features(titanic_train1)
print(cat_features)
cont_features = utils.get_continuous_features(titanic_train1)
print(cont_features)

#handle missing data(imputation)
cat_imputers = utils.get_categorical_imputers(titanic_train1, cat_features)
titanic_train1[cat_features] = cat_imputers.transform(titanic_train1[cat_features])
cont_imputers = utils.get_continuous_imputers(titanic_train1, cont_features)
titanic_train1[cont_features] = cont_imputers.transform(titanic_train1[cont_features])

#one hot encoding
titanic_train2 = utils.ohe(titanic_train1, cat_features)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(titanic_train2)
y_train = titanic_train['Survived']

tpot_estimator = tpot.TPOTClassifier(generations=10, population_size=20, verbosity=2)
tpot_estimator.fit(X_train, y_train)
print(tpot_estimator.score(X_train, y_train))
print(tpot_estimator.fitted_pipeline_)
print(tpot_estimator.evaluated_individuals_)
tpot_estimator.export(os.path.join(dir, 'tpot_titanic_pipeline.py'))

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))

print(titanic_test.shape)
print(titanic_test.info())

titanic_test1 = utils.drop_features(titanic_test, ['PassengerId', 'Name', 'Ticket', 'Cabin'])

utils.cast_to_cat(titanic_test1, ['Sex', 'Pclass', 'Embarked'])

cat_features = utils.get_categorical_features(titanic_test1)
print(cat_features)
cont_features = utils.get_continuous_features(titanic_test1)
print(cont_features)

titanic_test1[cat_features] = cat_imputers.transform(titanic_test1[cat_features])
titanic_test1[cont_features] = cont_imputers.transform(titanic_test1[cont_features])

print(titanic_test1.info())

titanic_test2 = utils.ohe(titanic_test1, cat_features)
X_test = scaler.transform(titanic_test2)
titanic_test['Survived'] = tpot_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)