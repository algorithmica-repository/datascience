import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, neighbors, metrics
import numpy as np

X_train, y_train = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=500, noise=0.25)
cutils.plot_data_2d_classification(X_train, y_train)

#underfitted learning in knn
knn_estimator = neighbors.KNeighborsClassifier(n_neighbors=400)
knn_estimator.fit(X_train, y_train)
cv_scores = model_selection.cross_val_score(knn_estimator, X_train, y_train, cv= 10)
print(np.mean(cv_scores))
train_score = knn_estimator.score(X_train, y_train)
print(train_score)
cutils.plot_model_2d_classification(knn_estimator, X_train, y_train)

#underfitted learning in knn
knn_estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_estimator.fit(X_train, y_train)
cv_scores = model_selection.cross_val_score(knn_estimator, X_train, y_train, cv= 10)
print(np.mean(cv_scores))
train_score = knn_estimator.score(X_train, y_train)
print(train_score)
cutils.plot_model_2d_classification(knn_estimator, X_train, y_train)

#grid seach tuning
knn_estimator = neighbors.KNeighborsClassifier()
knn_grid  = {'n_neighbors':list(range(1,200)) }
cutils.grid_search_plot_one_parameter_curves(knn_estimator, knn_grid, X_train, y_train )
cutils.grid_search_plot_models_classification(knn_estimator, knn_grid, X_train, y_train)
final_estimator = cutils.grid_search_best_model(knn_estimator, knn_grid, X_train, y_train)
