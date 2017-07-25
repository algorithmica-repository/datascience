import os
import pandas as pd
from sklearn import ensemble
from sklearn import tree
from sklearn import model_selection

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

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier(random_state=2017)
rf = ensemble.RandomForestClassifier(random_state=2017)
adaboost = ensemble.AdaBoostClassifier(random_state=2017)

#key of each model must be used as parameter prefix
voting_grid = dict(rf__n_estimators=list(range(100,1000,100)), rf__max_features=list(range(3,8,1)),
               dt__max_depth=list(range(3,7)),
                ada__n_estimators=list(range(100,1000,100)), ada__learning_rate=[0.1,0.3,0.5])   
v_estimator1 = ensemble.VotingClassifier([('dt',dt), ('rf',rf), ('ada',adaboost)])
voting_grid_estimator = model_selection.GridSearchCV(estimator=v_estimator1, param_grid=voting_grid, cv=10)
voting_grid_estimator.fit(X_train,y_train)
