import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, metrics, neighbors

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=4, weights=[0.3,0.3,0.2,0.2])
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)
cutils.plot_data_2d_classification(X_test, y_test)

knn_estimator = neighbors.KNeighborsClassifier()
knn_estimator.fit(X_train, y_train)
cutils.plot_model_2d_classification(knn_estimator, X_train, y_train)

y_pred = knn_estimator.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
metrics.confusion_matrix(y_test, y_pred)
