import pandas as pd

print(pd.__version__)
titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
titanic_train.groupby('Survived').size()
titanic_train.groupby('Sex').size()

titanic_train.groupby(['Sex','Survived']).size()


titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
titanic_test['Survived'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Survived'] = 1
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)

