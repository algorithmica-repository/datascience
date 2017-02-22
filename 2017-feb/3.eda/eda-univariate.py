import os
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns

pd.__version__

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.describe()

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

titanic_train['SibSp'].describe()

#explore univariate continuous features visually
sns.boxplot(x='Fare',data=titanic_train)
sns.distplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], bins=20, rug=True, kde=False)
sns.distplot(titanic_train['Fare'], bins=100, kde=False)
sns.kdeplot(data=titanic_train['Fare'])
sns.kdeplot(data=titanic_train['Fare'], shade=True)

#explore univariate categorical feature
titanic_train['Survived'].describe()
titanic_train['Survived'].value_counts()
pd.crosstab(index=titanic_train["Survived"], columns="count")
pd.crosstab(index=titanic_train["Pclass"], columns="count")  
pd.crosstab(index=titanic_train["Sex"],  columns="count")

#explore univariate categorical features visually
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)


