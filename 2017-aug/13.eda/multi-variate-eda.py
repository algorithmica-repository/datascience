import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("D:/titanic")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

sns.FacetGrid(titanic_train, row="Sex", col="Pclass").map(sns.countplot, "Survived")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.kdeplot, "Fare")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.kdeplot, "Age")
sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(sns.kdeplot, "Age").add_legend()
