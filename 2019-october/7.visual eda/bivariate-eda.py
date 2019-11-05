import pandas as pd
import os
import seaborn as sns

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

#explore bivariate relationships: categorical vs categorical 
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6)
#sns.factorplot(x="Survived", hue="Sex", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: continuous  vs categorical
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, col="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Age").add_legend()

#explore bivariate relationships: continuous vs continuous 
sns.jointplot(x="Age", y="Fare", data=titanic_train)

features = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch', 'Fare','Age', 'Survived']
sns.pairplot(titanic_train[features])
