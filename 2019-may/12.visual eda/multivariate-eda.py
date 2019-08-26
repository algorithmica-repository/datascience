import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = 'E:\\'
titanic_train = pd.read_csv(os.path.join(path, 'train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

#understand grid cell formation
g = sns.FacetGrid(titanic_train, col="Sex") 
g.map(sns.kdeplot, "Age")
g = sns.FacetGrid(titanic_train, row="Sex") 
g.map(sns.kdeplot, "Fare")

#is age have an impact on survived for each sex group?
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex") 
g.map(sns.kdeplot, "Age")

#is age have an impact on survived for each pclass and sex group?
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age")

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age")

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(plt.scatter, "Fare", "Age")

sns.heatmap(titanic_train[['Fare','Age','Parch']])


