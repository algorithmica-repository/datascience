import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, ensemble, tree
import pydot
import io
import os
import pandas as pd

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.5, 0.5])
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
ax = cutils.plot_data_2d_classification(X_train, y_train)
cutils.plot_data_2d_classification(X_test, y_test, ax, marker='x', s=70, legend=False)

#grid search for parameter values
rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'criterion':['gini', 'entropy'], 'max_depth':list(range(1,9)), 'n_estimators':list(range(1,100,10)) }
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, cv=10, refit=True)
rf_grid_estimator.fit(X_train, y_train)

#explore the attributes
print(rf_grid_estimator.cv_results_.get('params'))
print(rf_grid_estimator.best_params_)
print(rf_grid_estimator.best_score_)
final_estimator = rf_grid_estimator.best_estimator_
print(final_estimator)
print(final_estimator.estimators_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

dir = 'E:/'

#visualize the deciion tree
for i, est in enumerate(final_estimator.estimators_):
    X_df = pd.DataFrame(X_train, columns=['X0', 'X1'])
    dot_data = io.StringIO() 
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_df.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(os.path.join(dir, "tree" + str(i) + ".pdf"))

y_pred = final_estimator.predict(X_test)

