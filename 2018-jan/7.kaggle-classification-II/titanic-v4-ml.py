import pandas as pd
import os
from sklearn import tree
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
y_train = titanic_train[['Survived']]

#create an instance of machine learning class 
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
#build model by invoking fit method
dt_estimator.fit(X_train, y_train)

#model  evaluation phase
scores = model_selection.cross_validate(dt_estimator, X_train, y_train, cv=10)
print(scores.get('test_score').mean())
print(scores.get('train_score').mean())

#read and explore test data
titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

#fill up missing value in fare column
titanic_test.loc[titanic_test['Fare'].isnull(), 'Fare'] = 0

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_test1 = pd.get_dummies(titanic_test, columns=features)
print(titanic_test1.shape)

#Drop features not useful for learning pattern
features_to_drop = ['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_test1.drop(features_to_drop, axis=1, inplace=True)

X_test = titanic_test1
titanic_test['Survived'] = dt_estimator.predict(titanic_test1)
titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)
