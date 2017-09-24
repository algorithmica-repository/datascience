import os
import pandas as pd
from sklearn.manifold import TSNE
import utilities as util
import numpy as np

#changes working directory
os.chdir("D:/titanic")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
X_train.shape

tsne = TSNE(perplexity=30.0, n_components=2, n_iter=10000)
titanic_2 = tsne.fit_transform(X_train)


util.plot_data(titanic_2, np.array(titanic_train1['Survived']))

