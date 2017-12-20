import pandas as pd
import os
from sklearn.externals import joblib

print(os.getcwd())
os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

titanic_test.loc[titanic_test['Fare'].isnull(), 'Fare'] = 0

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_test1 = pd.get_dummies(titanic_test, columns=features)
print(titanic_test1.shape)

#Drop features not useful for learning pattern
features_to_drop = ['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_test1.drop(features_to_drop, axis=1, inplace=True)

X_test = titanic_test1

dt_estimator = joblib.load('decision-tree-v1.pkl')
titanic_test1['Survived'] = dt_estimator.predict(titanic_test1)
