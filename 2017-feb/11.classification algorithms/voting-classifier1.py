import os
import pandas as pd
from sklearn import tree
from sklearn import naive_bayes
from sklearn import ensemble
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


dt = tree.DecisionTreeClassifier(random_state=10)
nb = naive_bayes.GaussianNB()
rf = ensemble.RandomForestClassifier(random_state=10)
adaboost = ensemble.AdaBoostClassifier(random_state=10)

v_estimator = ensemble.VotingClassifier([('dt',dt),('nb', nb), ('rf',rf), ('aboost',adaboost)])
model_selection.cross_val_score(v_estimator, X_train, y_train, cv=10).mean()
v_estimator.fit(X_train,y_train)