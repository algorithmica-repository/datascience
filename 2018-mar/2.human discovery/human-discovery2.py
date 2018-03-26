import pandas as pd

titanic_train = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_train.csv')
print(titanic_train.shape)
print(titanic_train.info())

titanic_train.groupby(['Survived','Sex']).count()

titanic_test = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_test.csv')
print(titanic_test.shape)
print(titanic_test.info())
#gender based prediction logic/model
titanic_test['Survived'] = 0
titanic_test.loc[titanic_test['Sex']=='female','Survived'] = 1

titanic_test.to_csv('C:/Users/Algorithmica/Downloads/submission.csv', columns=['PassengerId','Survived'], index=False)

