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

rf_estimator = ensemble.RandomForestClassifier()
rf_grid = {'n_estimators':[100,200], 'max_features':[2,3,4]}
grid_model = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='roc_auc',cv=10, n_jobs=4)
grid_model.fit(X_train,y_train)
grid_model.grid_scores_
grid_model.best_estimator_.feature_importances_
