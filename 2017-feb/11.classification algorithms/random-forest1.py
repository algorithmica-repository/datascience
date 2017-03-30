import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import model_selection

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

sum(titanic_train['Pclass'].isnull())
titanic_train.apply(lambda x : sum(x.isnull()))
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'

titanic_train1 = titanic_train.copy()
le = preprocessing.LabelEncoder()
titanic_train1.Sex = le.fit_transform(titanic_train1.Sex)
titanic_train1.Embarked = le.fit_transform(titanic_train.Embarked)
titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)

X_train = titanic_train1[['Sex','Embarked','Pclass','Fare']]
y_train = titanic_train1['Survived']

#oob scrore is computed as part of model construction process
rf_estimator1 = ensemble.RandomForestClassifier(n_estimators=100,oob_score=True, random_state=10)
rf_estimator1.fit(X_train,y_train)
rf_estimator1.oob_score_

#evalaution is done separately from model building
rf_estimator2 = ensemble.RandomForestClassifier(n_estimators=100,oob_score=False, random_state=10)
model_selection.cross_val_score(rf_estimator2, X_train, y_train).mean()
rf_estimator2.fit(X_train,y_train)

#CV-evalaution is done for each combination of parameters
#Final model is built based on best parameterd discovered in the process
rf_estimator3 = ensemble.RandomForestClassifier(oob_score=True)
param_grid = {'n_estimators':[100,200,500,1000], 'max_features':[2,3,4]}
grid_model = model_selection.GridSearchCV(rf_estimator3, param_grid, cv=10, n_jobs=4)
grid_model.fit(X_train,y_train)
grid_model.grid_scores_