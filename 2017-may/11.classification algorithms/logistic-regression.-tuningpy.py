import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection

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
X_train.info()
y_train = titanic_train['Survived']

lr_estimator = linear_model.LogisticRegression(random_state=2017)
lr_grid = {'C':list(np.arange(0.1,1.0,0.1)), 'penalty':['l1','l2'], 'max_iter':list(range(100,1000,200))}
lr_grid_estimator = model_selection.GridSearchCV(lr_estimator, lr_grid, cv=10, n_jobs=5)
lr_grid_estimator.fit(X_train, y_train)
lr_grid_estimator.grid_scores_
final_model = lr_grid_estimator.best_estimator_
lr_grid_estimator.best_score_
final_model.coef_
final_model.intercept_

