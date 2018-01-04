import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection

os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

features = ['SibSp']
X_train = titanic_train[features]
y_train = titanic_train[['Survived']]

dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':list(range(1,4))}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
grid_dt_estimator.fit(X_train, y_train)
print(grid_dt_estimator.best_estimator_)
print(grid_dt_estimator.best_params_)

#find the cv and train scores of final model
print(grid_dt_estimator.best_score_)
print(grid_dt_estimator.score(X_train, y_train))

