import pandas as pd
import os
from sklearn import naive_bayes
from sklearn import model_selection

os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_train1 = pd.get_dummies(titanic_train, columns=features)
print(titanic_train1.shape)

#Drop features not useful for learning pattern
features_to_drop = ['PassengerId', 'Survived', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_train1.drop(features_to_drop, axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']
X_train.info()

nb_estimator = naive_bayes.GaussianNB()
nb_estimator.fit(X_train, y_train)
result = model_selection.cross_validate(nb_estimator, X_train, y_train, cv=10)
print(result.get('test_score').mean())
print(result.get('train_score').mean())

print(nb_estimator.class_prior_)
print(nb_estimator.sigma_)
print(nb_estimator.theta_)
