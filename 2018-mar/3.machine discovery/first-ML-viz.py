import pandas as pd
import os
from sklearn import tree
import pydot
import io

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

features = ['Parch','SibSp']
X_train = titanic_train[features]
y_train = titanic_train[['Survived']]
classifer = tree.DecisionTreeClassifier()
#build decision tree model from given data
classifer.fit(X_train,y_train)

#visualize the deciion tree
print(os.getcwd())
dot_data = io.StringIO() 
tree.export_graphviz(classifer, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf(os.path.join(path, "tree.pdf"))

titanic_test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))
print(titanic_test.shape)
print(titanic_test.info())

X_test = titanic_test[features]
#predict the target value for each test sample using the model built by fit
titanic_test['Survived'] = classifer.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
