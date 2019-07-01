import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import neighbors, preprocessing, model_selection

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.5, 0.5])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

#fit must be called only on train data whereas transform will be called on both train and test data
scaled_tr = preprocessing.StandardScaler()
scaled_tr.fit(X_train)
print(scaled_tr.mean_)
print(scaled_tr.scale_)
X_train_scaled = scaled_tr.transform(X_train)

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid  = {'n_neighbors':list(range(1,21)), 'weights':['uniform', 'distance'] }
final_estimator = cutils.grid_search_best_model(knn_estimator, knn_grid, X_train_scaled, y_train)

X_test_scaled = scaled_tr.transform(X_test)
y_pred = final_estimator.predict(X_test_scaled)
