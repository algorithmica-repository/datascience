import os
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection

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

rf_estimator = ensemble.RandomForestClassifier(random_state=2017)     
gbm_estimator = ensemble.GradientBoostingClassifier(random_state=2017)
lr_estimator = linear_model.LogisticRegression(random_state=2017)     

            
hard_voting_estimator = ensemble.VotingClassifier(estimators=[('rf', rf_estimator), ('gbm', gbm_estimator),('lr', lr_estimator)], voting='hard')
voting_grid = {'rf__n_estimators':[100,200], 'rf__max_features':[2,3], 'gbm__n_estimators':[100,150], 'gbm__max_depth':[5], 'gbm__learning_rate':[0.05], 'lr__C':[0.4,0.1]}
grid1 = model_selection.GridSearchCV(estimator=hard_voting_estimator, param_grid= voting_grid, n_jobs=1, cv=5)
grid1.fit(X_train, y_train)
grid1.grid_scores_

soft_voting_estimator = ensemble.VotingClassifier(estimators=[('rf', rf_estimator), ('gbm', gbm_estimator),('lr', lr_estimator)], voting='soft', weights=[2,1,2])
grid2 = model_selection.GridSearchCV(estimator=soft_voting_estimator, param_grid= voting_grid, n_jobs=1, cv=5)
grid2.fit(X_train, y_train)
grid2.grid_scores_
