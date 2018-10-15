import pandas as pd

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

#analyze gender decides survived
titanic_train.groupby(['Sex', 'Survived']).size()
titanic_train.groupby(['Sex', 'Pclass', 'Survived']).size()
titanic_train.groupby(['Sex', 'Embarked', 'Survived']).size()
titanic_train['Fare'].describe()

titanic_train.groupby(['Fare']).size()


titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())
titanic_test['Survived'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Survived'] = 1
titanic_test.loc[titanic_test['Age'] <= 15, 'Survived'] = 1
titanic_test.loc[titanic_test['Fare'] >= 60, 'Survived'] = 1

titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)
