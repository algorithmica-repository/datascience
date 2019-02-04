import pandas as pd

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/titanic_train.csv")
print(titanic_train.shape)

print(titanic_train.info())

#discover pattern: which class is majority?
titanic_train.groupby('Survived').size()
titanic_train.groupby(['Sex','Survived']).size()
titanic_train.groupby(['Sex','Pclass','Survived']).size()
titanic_train.groupby(['Sex','Embarked','Survived']).size()

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/titanic_test.csv")
print(titanic_test.shape)

titanic_test['Survived'] = 0
titanic_test.loc[titanic_test.Sex =='female', 'Survived'] = 1
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/submission.csv", columns = ['PassengerId', 'Survived'], index=False)
