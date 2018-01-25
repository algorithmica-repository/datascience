import pandas as pd
import os
import numpy as np
from sklearn import tree
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
X_train = one_hot_encoder.transform(titanic_train1).toarray()
y_train = titanic_train[['Survived']]

dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'criterion':['gini','entropy'], 'max_depth':[3,4,5,6,7,8]}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10, refit='True', return_train_score=True)
grid_dt_estimator.fit(X_train, y_train)

print(grid_dt_estimator.best_estimator_)
print(grid_dt_estimator.best_params_)
print(grid_dt_estimator.best_score_)
print(grid_dt_estimator.score(X_train, y_train))

#read and explore test data
titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test[encodable_columns] = mapper.transform(titanic_test)
titanic_test1 = titanic_test.drop(['PassengerId', 'Name', 'Cabin','Ticket'], axis=1)
X_test = one_hot_encoder.transform(titanic_test1).toarray()

titanic_test['Survived'] = grid_dt_estimator.predict(X_test)
titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)
