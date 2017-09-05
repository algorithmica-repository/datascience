import pandas as pd
from sklearn import tree
from sklearn import model_selection
import os

os.chdir('D:/titanic')

titanic_train = pd.read_csv("train.csv")

#explore the dataframe
titanic_train.shape
titanic_train.info()

X_train = titanic_train[['Sex', 'Embarked', 'Pclass', 'SibSp']]
y_train = titanic_train['Survived']

tree_estimator = tree.DecisionTreeClassifier()
#resampling approach - cv
#10 models are built using 90% of train data
#average performance of each model is regarded as cv performance
#none of these 10 models are used as final model
model_selection.cross_val_score(tree_estimator, X_train, y_train, cv= 10).mean()
#build model using entire train data
tree_estimator.fit(X_train, y_train)
#re-substitution approach
tree_estimator.score(X_train, y_train)

#The performance of final model will be approximated by cv-performance



