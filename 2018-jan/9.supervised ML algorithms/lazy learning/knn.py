import pandas as pd
import os
import numpy as np
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import model_selection
from sklearn_pandas import DataFrameMapper,CategoricalImputer


os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

encodable_columns=['Sex', 'Embarked', 'Pclass']
feature_defs = [(col_name, preprocessing.LabelEncoder()) for col_name in encodable_columns]
mapper = DataFrameMapper(feature_defs)
mapper.fit(titanic_train)
titanic_train[encodable_columns] = mapper.transform(titanic_train)

titanic_train1 = titanic_train.drop(['PassengerId', 'Name', 'Cabin','Ticket','Survived'], axis=1)

one_hot_encoder = preprocessing.OneHotEncoder(categorical_features = np.array([0,1,6]))
one_hot_encoder.fit(titanic_train1)
print(one_hot_encoder.n_values_)
titanic_train2 = one_hot_encoder.transform(titanic_train1).toarray()

scaler = preprocessing.StandardScaler()
scaler.fit(titanic_train2)
X_train = scaler.transform(titanic_train2)
y_train = titanic_train[['Survived']]

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':list(range(1,11))}
grid_knn_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, cv=10, refit='True', return_train_score=True)
grid_knn_estimator.fit(X_train, y_train)

print(grid_knn_estimator.best_params_)
print(grid_knn_estimator.best_score_)
print(grid_knn_estimator.score(X_train, y_train))

