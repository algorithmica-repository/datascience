import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, tree, neighbors
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

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid  = {'n_neighbors':list(range(1,21)), 'weights':['uniform', 'distance'] }
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, cv=10, refit=True)
knn_grid_estimator.fit(X_train, y_train)

print(knn_grid_estimator.cv_results_.get('params'))
print(knn_grid_estimator.best_params_)
print(knn_grid_estimator.best_score_)
final_estimator = knn_grid_estimator.best_estimator_
print(final_estimator)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

y_pred = final_estimator.predict(X_test)

