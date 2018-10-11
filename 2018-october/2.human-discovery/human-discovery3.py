import pandas as pd

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

#analyze gender decides survived
titanic_train.groupby(['Sex', 'Survived']).size()
titanic_train.groupby(['Pclass', 'Survived']).size()


titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())
titanic_test['Survived'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Survived'] = 1
titanic_test.loc[titanic_test['Pclass'] == 1, 'Survived'] = 1

titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)

