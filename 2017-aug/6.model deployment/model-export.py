import pandas as pd
from sklearn import tree
from sklearn import model_selection
import os
from sklearn.externals import joblib

os.chdir('E:/dl')

titanic_train = pd.read_csv("train.csv")

#explore the dataframe
titanic_train.shape
titanic_train.info()

X_train = titanic_train[['Pclass', 'SibSp']]
y_train = titanic_train['Survived']

tree_estimator = tree.DecisionTreeClassifier()
model_selection.cross_val_score(tree_estimator, X_train, y_train, cv= 10).mean()
tree_estimator.fit(X_train, y_train)
tree_estimator.score(X_train, y_train)

joblib.dump(tree_estimator, "dt-v1.pkl")


