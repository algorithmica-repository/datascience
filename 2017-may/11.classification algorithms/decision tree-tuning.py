import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
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

#automate model tuning process. use grid search method
dt = tree.DecisionTreeClassifier()
param_grid = {'criterion':['entropy'],'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[7,8,9,10,11,12]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=5)
dt_grid.fit(X_train, y_train)
dt_grid.grid_scores_
final_model = dt_grid.best_estimator_
dt_grid.best_score_
dt_grid.score(X_train, y_train)


dot_data = io.StringIO() 
tree.export_graphviz(final_model, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decisiont-tree-tuned1.pdf")