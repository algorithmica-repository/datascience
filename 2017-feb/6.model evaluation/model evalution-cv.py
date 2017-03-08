import os
import pandas as pd
from sklearn import tree
from sklearn.metrics import cross_val_score

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

X_train = titanic_train[['Fare']]
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()

#estimate the bias and variance of the model with cross validation
scores = cross_val_score(dt, X_train, y_train, cv = 5)
scores.mean()
scores.std()

#build the tree on entire train data
dt.fit(X_train,y_train)

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
X_test = titanic_test[['Fare']]
titanic_test['Survived'] = dt.predict_proba(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)