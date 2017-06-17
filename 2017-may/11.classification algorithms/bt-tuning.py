import os
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#list(range(100,1000,100))
dt = tree.DecisionTreeClassifier(random_state=1017)
bt_estimator = ensemble.BaggingClassifier(dt)
bt_grid = {'n_estimators':list(range(50,500,50))}
bt_grid_estimator = model_selection.GridSearchCV(bt_estimator,bt_grid, cv=10, n_jobs=10)
bt_grid_estimator.fit(X_train, y_train)
bt_grid_estimator.grid_scores_
bt_grid_estimator.best_score_

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = rf_grid_estimator.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)