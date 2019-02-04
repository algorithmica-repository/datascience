import pandas as pd
import seaborn as sns

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/titanic_train.csv")
print(titanic_train.shape)

print(titanic_train.info())

#visual discovery of  pattern

##univariate plots

#categorical columns: count/bar plot
#x: categories of feature, y: frequency
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)
sns.countplot(x='Sex',data=titanic_train)

#continuous columns: histogram/density plot/box-whisker plot
#x: bins of continuous data, y: frequency
#issue: how do you select number of bins?
sns.distplot(titanic_train['Fare'], kde=False)
sns.distplot(titanic_train['Fare'], bins=10, kde=False)
sns.distplot(titanic_train['Fare'], bins=100, kde=False)
#density plot to understand continuous feature
#it doesnt require bins argument
#x: fare y:density
sns.distplot(titanic_train['Fare'], hist=False)
sns.distplot(titanic_train['Fare'])
#box-whisker plot to understand continuous feature
sns.boxplot(x='Fare',data=titanic_train)

##Bi-variate plots
#category vs category: factor plot
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#continuous vs categorical: facet grid
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Age").add_legend()

#continuous vs continuous: scatter plot
sns.jointplot(x="Fare", y="Age", data=titanic_train)

##multi-variate plots

##multi-variate plots
#3-categorical features
g = sns.FacetGrid(titanic_train, row="Sex", col="Pclass") 
g.map(sns.countplot, "Survived")
g = sns.FacetGrid(titanic_train, row="Sex", col="Embarked") 
g.map(sns.countplot, "Survived")

#2-categorical variables & 1-continuous
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Fare")

#is age have an impact on survived for each pclass and sex group?
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age")

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/titanic_test.csv")
print(titanic_test.shape)

titanic_test['Survived'] = 0
titanic_test.loc[titanic_test.Sex =='female', 'Survived'] = 1
titanic_test.loc[( (titanic_test.Sex =='male') & (titanic_test.Pclass==1) ), 'Survived'] = 1

titanic_test.to_csv("C:/Users/Algorithmica/Downloads/submission.csv", columns = ['PassengerId', 'Survived'], index=False)
