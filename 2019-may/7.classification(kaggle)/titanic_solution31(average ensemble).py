import sys
sys.path.append("G:/")

import pandas as pd
import numpy as np
import os
import common_utils as utils
from sklearn import preprocessing, neighbors, svm, linear_model, ensemble, pipeline, model_selection
import classification_utils as cutils
import kernel_utils as kutils

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

#build model
knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = { 'n_neighbors': list(range(1,10))  }
knn_final_estimator =  cutils.grid_search_best_model(knn_estimator, knn_grid, X_train, y_train)

kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.001, 0.01, 0.1, 1], 'C':[0.001, 0.01, 1, 10, 100] }
svm_final_estimator = cutils.grid_search_best_model(kernel_svm_estimator, kernel_svm_grid, X_train, y_train)

stages = [('features', kutils.KernelTransformer('rbf')) ,
          ('clf', linear_model.LogisticRegression())
          ]
lr_pipeline = pipeline.Pipeline(stages)
lr_pipeline_grid  = {'features__gamma':[0.001, 0.01, 0.1, 1], 'clf__C':[0.001, 0.01, 1, 10, 100]}
lr_final_pipeline_estimator = cutils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train)

rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(5,10)), 'n_estimators':list(range(1,500, 100)) }
rf_final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, X_train, y_train)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,500, 100)), 'learning_rate':[0.1, 0.2, 0.5, 1.0] }
gb_final_estimator = cutils.grid_search_best_model(gb_estimator, gb_grid, X_train, y_train)


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

clfs = [knn_final_estimator, rf_final_estimator, lr_final_pipeline_estimator, svm_final_estimator, gb_final_estimator ]

def predict(X, clfs):
    classes = np.asarray([clf.predict(X) for clf in clfs])
    maj = np.asarray([np.argmax(np.bincount(classes[:,c])) for c in range(classes.shape[1])])
    return maj

titanic_test['Survived'] = predict(X_test, clfs)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)