import os
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import model_selection
import pydot
import io


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


#build boosted trees using adaboost algorithm
adaboost_estimator1 = ensemble.AdaBoostClassifier(n_estimators=5, random_state=100)
model_selection.cross_val_score(adaboost_estimator1, X_train, y_train, cv=10).mean()
adaboost_estimator1.fit(X_train,y_train)
adaboost_estimator1.estimator_errors_

#build boosted trees using adaboost algorithm with customization of decision trees
adaboost_estimator2 = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=3),n_estimators=5, random_state=100)
model_selection.cross_val_score(adaboost_estimator2, X_train, y_train, cv=10).mean()
adaboost_estimator2.fit(X_train,y_train)