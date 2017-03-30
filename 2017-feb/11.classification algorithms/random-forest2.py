import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
import pydot
from sklearn import tree
import io

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

#oob scrore is computed as part of model construction process
rf_estimator = ensemble.RandomForestClassifier(n_estimators=5,oob_score=True, max_features=2, random_state=10)
rf_estimator.fit(X_train,y_train)
rf_estimator.oob_score_

#extracting all the trees build by random forest algorithm
n_tree = 0
for est in rf_estimator.estimators_: 
    dot_data = io.StringIO()
    tmp = est.tree_
    tree.export_graphviz(tmp, out_file = dot_data, feature_names = X_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf("tree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1
