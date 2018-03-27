import pandas as pd
import os

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

titanic_train.groupby(['Sex','Survived']).size()
titanic_train.groupby(['Pclass','Survived']).size()
titanic_train.groupby(['Embarked','Survived']).size()
titanic_train.groupby(['Sex','Embarked','Survived']).size()

df1 = titanic_train[(titanic_train['Sex']=='female') & (titanic_train['Embarked']=='S')]
df1.groupby(['Pclass','Survived']).size()

df2 = titanic_train[(titanic_train['Sex']=='male') & (titanic_train['Pclass']==1)]
df2.groupby(['Embarked','Survived']).size()

titanic_test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))
print(titanic_test.shape)
print(titanic_test.info())

titanic_test['Survived'] = 0
titanic_test.loc[titanic_test['Sex']=='female','Survived'] = 1
titanic_test.loc[(titanic_test['Sex']=='female') & (titanic_test['Embarked']=='S') & (titanic_test['Pclass']==3),'Survived'] = 0
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)

