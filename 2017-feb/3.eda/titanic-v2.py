import os
import pandas as pd

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()
titanic_train.describe()
#titanic_train['Survived'].value_counts()
titanic_train.groupby('Survived').size()

titanic_train.groupby(['Sex','Survived']).size()
titanic_train.groupby(['Pclass','Survived']).size()
titanic_train.groupby(['Pclass','Sex','Survived']).size()
titanic_train.groupby(['Sex','Pclass','Survived']).size()
titanic_train.groupby(['Embarked','Survived']).size()
titanic_train.groupby(['Sex','Embarked','Survived']).size()



titanic_train.groupby('Fare').size()