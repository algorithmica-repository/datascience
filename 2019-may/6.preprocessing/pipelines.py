import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import neighbors, preprocessing, model_selection, pipeline

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.5, 0.5])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

#pipelines without gridsearch
stages = [
            ('scaling', preprocessing.StandardScaler()),
            ('classifier', neighbors.KNeighborsClassifier())
        ]
knn_pipeline = pipeline.Pipeline(stages)
knn_pipeline.fit(X_train, y_train)

y_pred = knn_pipeline.predict(X_test)

#pipelines with gridsearch
knn_estimator = neighbors.KNeighborsClassifier()
knn_grid  = {'n_neighbors':list(range(1,21)), 'weights':['uniform', 'distance'] }
stages = [
            ('scaling', preprocessing.StandardScaler()),
            ('grid_classifier', model_selection.GridSearchCV(knn_estimator, knn_grid))
        ]
knn_grid_pipeline = pipeline.Pipeline(stages)
knn_grid_pipeline.fit(X_train, y_train)
knn_grid_estimator = knn_grid_pipeline.named_steps['grid_classifier']
print(knn_grid_estimator.best_estimator_)

y_pred = knn_grid_pipeline.predict(X_test)
