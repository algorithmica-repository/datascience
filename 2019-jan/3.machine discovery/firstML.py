import pandas as pd
from sklearn import tree
import pydot
import io


#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

features = ['Pclass', 'Parch' , 'SibSp']
X_train = titanic_train[features]
y_train = titanic_train['Survived']
#create an instance of decision tree classifier type
classifer = tree.DecisionTreeClassifier()
print(type(classifer))

#learn the pattern automatically
classifer.fit(X_train, y_train)

#get the logic or model learned by Algorithm
#issue: not readable
print(classifer.tree_)

#get the readable tree structure from tree_ object
#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(classifer, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("C:/Users/Algorithmica/Downloads/tree.pdf")

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())
X_test = titanic_test[features]
titanic_test['Survived'] = classifer.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)
