import pandas as pd
import os
import seaborn as sns

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

#categorical columns: numerical EDA
pd.crosstab(index=titanic_train["Survived"], columns="count")

#categorical columns: visual EDA
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)
sns.countplot(x='Sex',data=titanic_train)

#continuous features: visual EDA
titanic_train['Fare'].describe()
sns.boxplot(x='Fare',data=titanic_train)
sns.distplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], hist=False)
sns.distplot(titanic_train['Age'], hist=False)
sns.boxplot(x='Age',data=titanic_train)

sns.distplot(titanic_train['SibSp'], hist=False)
sns.boxplot(x='SibSp',data=titanic_train)
