import sys
sys.path.append("E:/")

import pandas as pd
import os
import common_utils as utils
from sklearn import preprocessing, neighbors, svm, linear_model, ensemble, pipeline, model_selection
import classification_utils as cutils
import kernel_utils as kutils

dir = 'E:/'
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

#adding new levels
#titanic_train['Pclass'] = titanic_train['Pclass'].cat.add_categories([4,5])

#one hot encoding
titanic_train2 = utils.ohe(titanic_train1, cat_features)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(titanic_train2)
y_train = titanic_train['Survived']

#build model
knn_estimator = neighbors.KNeighborsClassifier()
kernel_svm_estimator = svm.SVC(kernel='rbf')
stages = [('features', kutils.KernelTransformer('rbf')) ,
          ('clf', linear_model.LogisticRegression())
          ]
lr_pipeline = pipeline.Pipeline(stages)
rf_estimator = ensemble.RandomForestClassifier()
gb_estimator = ensemble.GradientBoostingClassifier()

estimators = [
        ('gb', gb_estimator) , 
        ('knn', knn_estimator), 
        ('rf', rf_estimator),
        ('svm', kernel_svm_estimator),
        ('lr', lr_pipeline)
        ]

hvoting_estimator = ensemble.VotingClassifier(estimators)
hvoting_grid = {
        'gb__max_depth':[2], 'gb__n_estimators':list(range(300,500, 100)), 'gb__learning_rate':[0.1, 0.2, 0.5, 1.0], 
        'knn__n_neighbors': list(range(6,10)),
        'rf__max_depth':list(range(6,8)), 'rf__n_estimators':list(range(200,400, 100)),
         'lr__features__gamma':[0.001, 0.01], 'lr__clf__C':[1, 10],
         'svm__gamma':[0.001, 0.01], 'svm__C':[0.001, 0.01, 1, 10]         
        }
voting_final_estimator = cutils.grid_search_best_model(hvoting_estimator, hvoting_grid, X_train, y_train)


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
titanic_test['Survived'] = voting_final_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
