import pandas as pd
import os
from sklearn import neighbors, model_selection, preprocessing

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

lencoder = preprocessing.LabelEncoder()
lencoder.fit(titanic_train['Sex'])
print(lencoder.classes_)
titanic_train['Sex_encoded'] = lencoder.transform(titanic_train['Sex'])

imputer = preprocessing.Imputer()
imputer.fit(titanic_train[['Age']])
print(imputer.statistics_)
titanic_train['Age_imputed'] = imputer.transform(titanic_train[['Age']])

features = ['SibSp', 'Parch', 'Pclass', 'Sex_encoded', 'Age_imputed']
X = titanic_train[ features ]
y = titanic_train['Survived']

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

#grid search based model building
knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors': list(range(2,10)) }
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, scoring='accuracy', cv=10)
knn_grid_estimator.fit(X_train, y_train)
print(knn_grid_estimator.best_params_)
print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.score(X_train, y_train))

print(knn_grid_estimator.score(X_eval, y_eval))

