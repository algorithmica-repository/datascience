import os
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib

os.chdir("C:\\Users\\Algorithmica\\Downloads")

dt = joblib.load("dt1.pkl")

#pipeline for test data
titanic_test = pd.read_csv("test.csv")
titanic_test.apply(lambda x : sum(x.isnull()))
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = titanic_test.copy()
le = preprocessing.LabelEncoder()
titanic_test1.Sex = le.fit_transform(titanic_test1.Sex)
titanic_test1.Embarked = le.fit_transform(titanic_test1.Embarked)
titanic_test1.Pclass = le.fit_transform(titanic_test1.Pclass)

X_test = titanic_test1[['Sex','Embarked','Pclass','Fare']]
titanic_test1['Survived'] = dt.predict(X_test)

titanic_test1.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)