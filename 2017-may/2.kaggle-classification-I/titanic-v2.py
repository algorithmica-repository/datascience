import pandas as pd
import os

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("/home/algo/Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.groupby('Survived').size()
titanic_train.groupby(['Sex','Survived']).size()

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.info()
titanic_test['Survived'] = 0
titanic_test.loc[titanic_test.Sex == 'female','Survived'] = 1
print(titanic_test[['PassengerId','Survived']])
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)