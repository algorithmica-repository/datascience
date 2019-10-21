import pandas as pd
import os
from sklearn import tree, model_selection
import io
import pydot

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

X_train = titanic_train[ ['SibSp', 'Parch'] ]
y_train = titanic_train['Survived']
dt_estimator = tree.DecisionTreeClassifier()
dt_estimator.fit(X_train, y_train)
print(dt_estimator.tree_)
model_selection.cross_val_score(dt_estimator, X_train, y_train, scoring="accuracy", cv=5).mean()

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
dir = 'E:/'
graph.write_pdf(os.path.join(dir, "tree.pdf"))

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.info())
X_test = titanic_test[ ['SibSp', 'Parch'] ]
titanic_test['Survived'] = dt_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
