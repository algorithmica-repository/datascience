import os
import pandas as pd

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()
#titanic_train.describe()
#titanic_train['Survived'].value_counts()
titanic_train.groupby('Survived').size()

titanic_train.groupby(['Sex','Survived']).size()

titanic_test = pd.read_csv("test.csv")


titanic_test.shape
titanic_test.info()
titanic_test['Survived'] = 0
titanic_test.Survived[titanic_test.Sex == "female"] = 1
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)