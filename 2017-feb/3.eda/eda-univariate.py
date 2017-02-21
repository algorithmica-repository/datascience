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
titanic_train.describe()

type(titanic_train['Fare'])

titanic_train['Survived'] = titanic_train['Survived'].astype('category')
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category')
titanic_train['Sex'] = titanic_train['Sex'].astype('category')
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category')

titanic_train.describe()

#explore univariate continuous feature
print titanic_train['Fare'].mean()
print titanic_train['Fare'].median()
print titanic_train['Fare'].quantile(0.25)
print titanic_train['Fare'].quantile(0.75)
print titanic_train['Fare'].std()
titanic_train['Fare'].describe()

#explore univariate continuous features visually

#explore univariate categorical feature
titanic_train['Survived'].describe()
titanic_train['Pclass'].describe()

#explore univariate categorical features visually
