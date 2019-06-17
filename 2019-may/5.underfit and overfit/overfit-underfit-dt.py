import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, tree
import pydot
import io
import os
import pandas as pd
import numpy as np

X_train, y_train = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=500, noise=0.25)
cutils.plot_data_2d_classification(X_train, y_train)

#underfitted learning in dt
dt_estimator = tree.DecisionTreeClassifier(max_depth=1)
dt_estimator.fit(X_train, y_train)
cv_scores = model_selection.cross_val_score(dt_estimator, X_train, y_train, cv= 10)
print(np.mean(cv_scores))
train_score = dt_estimator.score(X_train, y_train)
print(train_score)

#visualize the deciion tree
X_df = pd.DataFrame(X_train, columns=['X0', 'X1'])
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_df.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
dir = 'E:/'
graph.write_pdf(os.path.join(dir, "tree.pdf"))

#overfitted learning in dt
dt_estimator = tree.DecisionTreeClassifier(max_depth=15)
dt_estimator.fit(X_train, y_train)
cv_scores = model_selection.cross_val_score(dt_estimator, X_train, y_train, cv= 10)
print(np.mean(cv_scores))
train_score = dt_estimator.score(X_train, y_train)
print(train_score)

#visualize the deciion tree
X_df = pd.DataFrame(X_train, columns=['X0', 'X1'])
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_df.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
dir = 'E:/'
graph.write_pdf(os.path.join(dir, "tree.pdf"))

dt_estimator = tree.DecisionTreeClassifier()
dt_grid  = {'max_depth':list(range(1,9)) }
cutils.grid_search_plot_one_parameter_curves(dt_estimator, dt_grid, X_train, y_train )
cutils.grid_search_plot_models_classification(dt_estimator, dt_grid, X_train, y_train)
final_estimator = cutils.grid_search_best_model(dt_estimator, dt_grid, X_train, y_train)

cutils.generate_linear_synthetic_data_classification
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10, refit=True)
dt_grid_estimator.fit(X_train, y_train)