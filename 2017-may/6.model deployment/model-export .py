import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn.externals import joblib

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#data preparation
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

#feature engineering 
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
#use cross validation to estimate performance of model. 
#No model build during cross validation is not used as final model
cv_scores = model_selection.cross_val_score(dt, X_train, y_train, cv=10, verbose=1)
cv_scores.mean()

#build final model on entire train data which is used for prediction
dt.fit(X_train,y_train)

# natively deploy decision tree model(pickle format)
joblib.dump(dt, "tree1.pkl")