import os
import pandas as pd
from sklearn import neighbors
from sklearn import model_selection
from sklearn import preprocessing

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
scaler = preprocessing.StandardScaler()
X_train1 = scaler.fit_transform(X_train)
y_train = titanic_train['Survived']

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':list(range(3,20))}
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator,knn_grid, cv=10, n_jobs=10)
knn_grid_estimator.fit(X_train1, y_train)
knn_grid_estimator.grid_scores_
knn_grid_estimator.best_estimator_
knn_grid_estimator.best_score_
knn_grid_estimator.best_estimator_.feature_importances_

knn_grid_estimator.predict()