import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection

os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_train1 = pd.get_dummies(titanic_train, columns=features)
print(titanic_train1.shape)

#Drop features not useful for learning pattern
features_to_drop = ['PassengerId', 'Survived', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_train1.drop(features_to_drop, axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train[['Survived']]

#create an instance of machine learning class 
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_param_distr = {'criterion':['gini','entropy'], 'max_depth':[3,4,5,6,7], 'min_samples_split':[2,5,10]}
#create an instance of GridSearchCV class
random_dt_estimator = model_selection.RandomizedSearchCV(dt_estimator, dt_param_distr, cv=10, n_iter= 12, refit='True', return_train_score=True)
#apply cv for each parameter combination and build final model on entire
#data with the best parameter combination
random_dt_estimator.fit(X_train, y_train)
#retrieve the final model with the best parameters
print(random_dt_estimator.best_estimator_)
print(random_dt_estimator.best_params_)
print(random_dt_estimator.best_score_)

#retrieve entire grid summary
print(len(random_dt_estimator.grid_scores_))
print(random_dt_estimator.cv_results_)

