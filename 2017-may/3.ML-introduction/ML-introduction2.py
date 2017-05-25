import os
import pandas as pd
from sklearn import tree
import io
import pydot

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("/home/algo/Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

X_train = titanic_train[['Pclass']]
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("dt2.pdf")

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
X_test = titanic_test[['Pclass']]
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)