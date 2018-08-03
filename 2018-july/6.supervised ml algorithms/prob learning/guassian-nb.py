import pandas as pd
from sklearn import naive_bayes, model_selection

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

titanic_train.loc[titanic_train['Age'].isnull() == True, 'Age'] = titanic_train['Age'].mean()

cat_columns = ['Sex', 'Embarked', 'Pclass']
titanic_train1 = pd.get_dummies(titanic_train, columns = cat_columns)
titanic_train1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']

classifier = naive_bayes.GaussianNB()
res = model_selection.cross_validate(classifier, X_train, y_train, cv=10)
print(res.get('test_score').mean())
classifier.fit(X_train, y_train)
print(classifier.class_prior_)
print(classifier.sigma_)
print(classifier.theta_)

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
print(titanic_test.info())

titanic_test.loc[titanic_test['Age'].isnull() == True, 'Age'] = titanic_test['Age'].mean()
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()
titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
titanic_test1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test = titanic_test1
titanic_test['Survived'] = classifier.predict(X_test)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)
