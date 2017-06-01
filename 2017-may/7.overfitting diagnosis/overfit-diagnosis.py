import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

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

#create the decision tree model object
dt = tree.DecisionTreeClassifier()

#compute the cv accuracy
cv_scores = model_selection.cross_val_score(dt, X_train, y_train, cv=10)
cv_scores.mean()

#build model
dt.fit(X_train,y_train)

#compute train accuracy: approach1
dt.score(X_train,y_train)

#compute train accuracy: approach2
y_pred = dt.predict(X_train)
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
metrics.accuracy_score(y_train, y_pred)

