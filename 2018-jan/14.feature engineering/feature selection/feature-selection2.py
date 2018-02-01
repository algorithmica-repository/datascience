import pandas as pd
import os
from sklearn import tree
from sklearn import preprocessing
from sklearn import model_selection
from sklearn_pandas import DataFrameMapper
from sklearn import feature_selection

os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

titanic_train.loc[titanic_train['Embarked'].isnull(), 'Embarked'] = 'S'

encodable_columns=['Sex', 'Embarked', 'Pclass']
feature_defs = [(col_name, preprocessing.LabelEncoder()) for col_name in encodable_columns]
mapper = DataFrameMapper(feature_defs)
mapper.fit(titanic_train)
titanic_train[encodable_columns] = mapper.transform(titanic_train)

titanic_train1 = titanic_train.drop(['PassengerId', 'Name', 'Cabin','Ticket','Survived'], axis=1)


features = ['Pclass', 'Sex', 'Embarked']
titanic_train2 = pd.get_dummies(titanic_train1, columns=features)

y_train = titanic_train['Survived']
X_train = titanic_train2

dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'criterion':['gini','entropy'], 'max_depth':[3,4,5,6,7,8]}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
grid_dt_estimator.fit(X_train, y_train)

best_est = grid_dt_estimator.best_estimator_
print(best_est.feature_importances_)

model = feature_selection.SelectFromModel(best_est, prefit=True)
X_new = model.transform(X_train)
X_new.shape 

#build model on X_new