import os
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns

#changes working directory
os.chdir("D:/titanic")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#bivariate relationships(c-c): statistical EDA 
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)

sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#bivariate relationships(n-c): visual EDA 
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()

titanic_train.loc[titanic_train['Age'].isnull() == True, 'Age'] = titanic_train['Age'].mean()

sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Age").add_legend()

