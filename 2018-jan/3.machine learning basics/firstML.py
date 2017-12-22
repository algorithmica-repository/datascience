import pandas as pd
import os
from sklearn import tree
import pydot
import io

print(os.getcwd())
os.chdir('C:/Users/Algorithmica/Downloads')

titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#separeate features from target column
features = ['Pclass']
X_train = titanic_train[features]
y_train = titanic_train[['Survived']]

#create an instance of machine learning class 
dt_estimator = tree.DecisionTreeClassifier()
#build model by invoking fit method
dt_estimator.fit(X_train, y_train)

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decision-tree.pdf")

titanic_test = pd.read_csv('titanic_test.csv')
titanic_test.shape
titanic_test.info()

X_test = titanic_test[features]
titanic_test['Survived'] = dt_estimator.predict(X_test)
titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)
