import os
import pandas as pd
from sklearn import naive_bayes
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

nb_estimator = naive_bayes.BernoulliNB()
scores = model_selection.cross_val_score(nb_estimator, X_train, y_train, cv = 10)
scores.mean()
nb_estimator.fit(X_train, y_train)
