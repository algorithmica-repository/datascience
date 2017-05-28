import os
import pandas as pd
from sklearn import tree

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("/home/algo/Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

X_train = titanic_train[['Pclass']]
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
X_test = titanic_test[['Pclass']]
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)