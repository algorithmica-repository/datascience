import os
import pandas as pd
from sklearn import tree
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import pydot
import io
import numpy as np

sklearn.__version__
#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

sum(titanic_train['Pclass'].isnull())
titanic_train.apply(lambda x : sum(x.isnull()))
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'

titanic_train1 = titanic_train.copy()
le = preprocessing.LabelEncoder()
titanic_train1.Sex = le.fit_transform(titanic_train1.Sex)
titanic_train1.Embarked = le.fit_transform(titanic_train.Embarked)
titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)

X_train = titanic_train1[['Sex','Embarked','Pclass','Fare']]
y_train = titanic_train1['Survived']

dt_estimator = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12]}
dt_grid = model_selection.GridSearchCV(dt_estimator, param_grid, cv=10, n_jobs=5)

#build model using entire train data
dt_grid.fit(X_train,y_train)

dt_grid.grid_scores_
dt_grid.best_params_
dt_grid.best_score_

dot_data = io.StringIO() 
tree.export_graphviz(dt_grid, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("dt.pdf")



#pipeline for test data
titanic_test = pd.read_csv("test.csv")
titanic_test.apply(lambda x : sum(x.isnull()))
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()


titanic_test1 = titanic_test.copy()
titanic_test1.Sex = le.fit_transform(titanic_test1.Sex)
titanic_test1.Embarked = le.fit_transform(titanic_test1.Embarked)
titanic_test1.Pclass = le.fit_transform(titanic_test1.Pclass)

X_test = titanic_test1[['Sex','Embarked','Pclass','Fare']]
titanic_test1['Survived'] = dt.predict(X_test)

titanic_test1.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)