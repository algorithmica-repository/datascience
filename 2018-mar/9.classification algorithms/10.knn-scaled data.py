import pandas as pd
import os
from sklearn import model_selection, neighbors
from sklearn import preprocessing

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

features = ['Sex', 'Pclass', 'Embarked','Parch','SibSp']
titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Survived','Name','Age','Cabin','Ticket'], axis=1)
y_train = titanic_train['Survived']

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train1 = scaler.transform(X_train)

classifier = neighbors.KNeighborsClassifier(n_jobs=1)
knn_grid = {'n_neighbors':list(range(1,20,2)), 'weights':['uniform','distance']}
grid_classifier = model_selection.GridSearchCV(classifier, knn_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train1, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_

titanic_test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))
print(titanic_test.shape)
print(titanic_test.info())
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Sex','Pclass','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Name','Age','Cabin','Ticket'], axis=1)
titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
