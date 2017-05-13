import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from stacking import StackEnsemble
from sklearn import linear_model

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

sum(titanic_train['Pclass'].isnull())
titanic_train.apply(lambda x : sum(x.isnull()))
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'

titanic_train1 = titanic_train.copy()
le = preprocessing.LabelEncoder()
titanic_train1.Sex = le.fit_transform(titanic_train1.Sex)
titanic_train1.Embarked = le.fit_transform(titanic_train.Embarked)
titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)

X_train = titanic_train1[['Sex','Embarked','Pclass','Fare']]
y_train = titanic_train1['Survived']

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

stacked_model = StackEnsemble(n_folds=5,base_models=base_models,
        stacker=ensemble.GradientBoostingClassifier(random_state=2017),
        stacker_grid = grid)

stacked_model.fit(X_train, y_train)
