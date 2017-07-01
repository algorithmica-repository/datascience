import os
os.chdir("E:\\")
import pandas as pd
from stacking import StackEnsemble
from sklearn import ensemble
from sklearn import linear_model

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

base_models = [
        ensemble.RandomForestClassifier(
            n_jobs=4, random_state=2017,
            n_estimators=100, max_features=3
        ),
        ensemble.GradientBoostingClassifier(
            random_state=2017,
            n_estimators=100, max_features=3, max_depth=5,
            learning_rate=0.05
       ),
       linear_model.LogisticRegression(
               random_state=2017,
               C=0.2)         
    ]
    
grid = {
            'n_estimators': [100,200],
            'learning_rate': [0.05],
            'max_depth':[4,5,6]
            }

stacked_model = StackEnsemble(n_folds=10,base_models=base_models,
        stacker=ensemble.GradientBoostingClassifier(random_state=2017),
        stacker_grid = grid)

stacked_model.fit(X_train, y_train)