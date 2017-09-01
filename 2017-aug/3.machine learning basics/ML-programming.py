import pandas as pd
from sklearn import tree
import pydot
import io
import os

os.chdir('D:/titanic')

titanic_train = pd.read_csv("train.csv")
print(type(titanic_train))

#explore the dataframe
titanic_train.shape
titanic_train.info()

X_train = titanic_train[['Pclass', 'Fare', 'SibSp']]
y_train = titanic_train['Survived']

tree_estimator = tree.DecisionTreeClassifier()
tree_estimator.fit(X_train, y_train)

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(tree_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decision-tree.pdf")
