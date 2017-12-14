import pandas as pd
import os

print(os.getcwd())
os.chdir('C:/Users/Algorithmica/Downloads')

titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

titanic_train.groupby(['Sex','Survived']).size()

titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

titanic_test['Survived'] = 0
titanic_test.loc[titanic_test.Sex=='female','Survived'] = 1
#titanic_test.loc[titanic_test.Sex=='male','Survived'] = 0
titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)


