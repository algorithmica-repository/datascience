import pandas as pd
import os
from sklearn import tree, model_selection
import pydot
import io

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

features = ['Sex', 'Pclass', 'Embarked','Parch','SibSp']
titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Survived','Name','Age','Cabin','Ticket'], axis=1)
y_train = titanic_train[['Survived']]

classifer = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[3,4,5,6], 'criterion':['gini','entropy']}
grid_classifier = model_selection.RandomizedSearchCV(classifer, dt_grid, n_iter=4, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_

