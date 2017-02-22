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

titanic_train['Survived'] = titanic_train['Survived'].astype('category')
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category')
titanic_train['Sex'] = titanic_train['Sex'].astype('category')
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category')


#explore bivariate relationships: categorical vs categorical 
tmp1 = pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
type(tmp1)
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)
