import pandas as pd
from sklearn import tree, model_selection
import pydot
import io

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

features = ['Pclass', 'Parch' , 'SibSp']
X_train = titanic_train[features]
y_train = titanic_train['Survived']

#create an estimator 
dt_estimator = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[3,4,5,6,7], 'criterion':['entropy','gini'] }
dt_grid_estimator = model_selection.RandomizedSearchCV(dt_estimator, dt_grid, n_iter=5, scoring='accuracy', cv=10, refit=True)
dt_grid_estimator.fit(X_train, y_train)

#explore the results of grid_search_cv estimator
print(dt_grid_estimator.cv_results_)
print(dt_grid_estimator.best_estimator_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_params_)

#visualuze the final model built with best parameters in grid
best_dt_estimator = dt_grid_estimator.best_estimator_
print(best_dt_estimator.tree_)
dot_data = io.StringIO() 
tree.export_graphviz(best_dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("C:/Users/Algorithmica/Downloads/tree2.pdf")


