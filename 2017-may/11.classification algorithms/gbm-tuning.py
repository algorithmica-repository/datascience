import os
import pandas as pd
from sklearn import ensemble
from sklearn import tree
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

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

gbm_tree_estimator1 = ensemble.GradientBoostingClassifier(n_estimators=5, random_state=2017)
gbm_grid = {'n_estimators':list(range(100,2000,100)),'learning_rate':[0.1,0.2, 0.3]}
gbm_grid_estimator = model_selection.GridSearchCV(gbm_tree_estimator1,gbm_grid, cv=10, n_jobs=10)
gbm_grid_estimator.fit(X_train, y_train)
gbm_grid_estimator.grid_scores_
gbm_grid_estimator.best_score_
gbm_grid_estimator.best_params_

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = gbm_grid_estimator.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
