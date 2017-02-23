import os
import pandas as pd
import seaborn as sns
import numpy as np

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
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)

sns.factorplot(x="Survived", hue="Sex", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: categorical vs continuous 
sns.factorplot(x="Fare", row="Survived", data=titanic_train, kind="box", size=6)

sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()

#explore bivariate relationships: continuous vs continuous 
np.cov(titanic_train['SibSp'], titanic_train['Parch'])
np.corrcoef(titanic_train['SibSp'], titanic_train['Parch'])
sns.jointplot(x="SibSp", y="Parch", data=titanic_train)
sns.jointplot(x="SibSp", y="Parch", data=titanic_train, kind='kde')

v1 = [10,12,15,20,22,25]
v2 = [20,22,25,27,32,35]
v3 = [35,32,30,29,27,26]
np.cov(v1,v2)
np.corrcoef(v1,v2)
df = pd.DataFrame({'v1':v1,'v2':v2,
'v3':v3})
sns.jointplot(x="v1", y="v3", data=df)

sns.pairplot(titanic_train)