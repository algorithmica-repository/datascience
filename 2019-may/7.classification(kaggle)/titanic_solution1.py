import sys
sys.path.append("E:/")

import pandas as pd
import os
import common_utils as utils
from sklearn import preprocessing, neighbors, svm, linear_model, ensemble, pipeline
import classification_utils as cutils

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
X_train = utils.ohe(titanic_train1, cat_features)
y_train = titanic_train['Survived']

#build model
knn_pipeline_stages = [
            ('scaler', preprocessing.StandardScaler()) ,
            ('knn', neighbors.KNeighborsClassifier())
          ]
knn_pipeline = pipeline.Pipeline(knn_pipeline_stages)
knn_pipeline_grid = { 'knn__n_neighbors': list(range(1,10))  }
knn_pipeline_model =  cutils.grid_search_best_model(knn_pipeline, knn_pipeline_grid, X_train, y_train)

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

X_test = utils.ohe(titanic_test1, cat_features)
titanic_test['Survived'] = knn_pipeline_model.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
