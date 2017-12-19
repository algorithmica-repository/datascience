import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection
import pydot
import io

print(os.getcwd())
os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_train1 = pd.get_dummies(titanic_train, columns=features)
print(titanic_train1.shape)

#Drop features not useful for learning pattern
features_to_drop = ['PassengerId', 'Survived', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_train1.drop(features_to_drop, axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train[['Survived']]

#create an instance of machine learning class 
dt_estimator = tree.DecisionTreeClassifier()
#build model by invoking fit method
dt_estimator.fit(X_train, y_train)

#model  evaluation phase
scores = model_selection.cross_validate(dt_estimator, X_train, y_train, cv=10)
scores.get('test_score').mean()
scores.get('train_score').mean()

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decision-tree.pdf")

#model deplyment phase
