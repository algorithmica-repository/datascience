import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#conditional plotting(factor plot and FacetGrid)
g = sns.FacetGrid(titanic_train, col="Sex") 
g.map(sns.distplot, "Age")

g = sns.FacetGrid(titanic_train, row="Sex") 
g.map(sns.distplot, "Fare")

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex") 
g.map(sns.distplot, "Age")
g.map(sns.kdeplot, "Age")

g = sns.FacetGrid(titanic_train, col="Survived") 
g.map(plt.scatter, "Parch", "SibSp")

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived")
g.map(sns.kdeplot, "Age")

plt.xlim(0, 250) 
plt.ylim(0, 60)

#interaction across each pair of features(pair plot and grid)
pairable_features = ["SibSp", "Parch", "Fare", "Age", "Survived"]
g = sns.PairGrid(titanic_train[pairable_features]) 
g.map_upper(sns.regplot) 
g.map_lower(sns.residplot) 
g.map_diag(sns.kdeplot)
g.add_legend() 

#joint distributions(joint plot and joint grid)
g = sns.JointGrid(x="SibSp", y="Parch", data=titanic_train) 
g.plot_joint(sns.regplot, order=2) 
g.plot_marginals(sns.distplot)
