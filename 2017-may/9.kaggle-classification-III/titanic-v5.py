import os
import pandas as pd
from sklearn import tree
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

dt = tree.DecisionTreeClassifier(random_state=2017)
param_grid = {'criterion':['entropy','gini'],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,20,25]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(X_train, y_train)
dt_grid.grid_scores_
dt_grid.best_estimator_
dt_grid.best_score_
dt_grid.score(X_train, y_train)

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = dt_grid.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
