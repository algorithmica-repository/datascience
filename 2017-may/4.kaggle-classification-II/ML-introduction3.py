import os
import pandas as pd
from sklearn import tree
import io
import pydot

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decisiont-tree.pdf")

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()


titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()
titanic_test1.head(6)

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)