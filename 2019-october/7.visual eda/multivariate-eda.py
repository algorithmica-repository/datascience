import pandas as pd
import os
import seaborn as sns

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

#understand grid cell formation
g = sns.FacetGrid(titanic_train, col="Sex") 
g.map(sns.kdeplot, "Age")
g.map(sns.boxplot, "Age")
g = sns.FacetGrid(titanic_train, row="Sex") 
g.map(sns.kdeplot, "Fare")
g = sns.FacetGrid(titanic_train, hue="Survived")
g.map(sns.kdeplot, "Fare").add_legend()

#is age have an impact on survived for each sex group?
g = sns.FacetGrid(titanic_train, col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age").add_legend()

#is age have an impact on survived for each pclass and sex group?
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age").add_legend()

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age")

tmp = titanic_train[['Fare','Age','Parch','SibSp']]
sns.heatmap(tmp.corr())
