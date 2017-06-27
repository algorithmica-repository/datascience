import os
import pandas as pd
import seaborn as sns
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

#explore missing data
titanic_train.apply(lambda x : sum(x.isnull()))

#pre-process Embarked
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'

#pre-process Age
imputer = preprocessing.Imputer()
titanic_train[['Age']] = imputer.fit_transform(titanic_train[['Age']])
sns.distplot(titanic_train['Age'])
sns.factorplot(x="Age", row="Survived", data=titanic_train, kind="box", size=6)
sns.factorplot(x="Age", row="Survived", data=titanic_train, kind="box", size=6)
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "Age").add_legend()

#create family size feature
def size_to_type(x):
    if(x == 1): 
        return 'Single'
    elif(x >= 2 and x <= 4): 
        return 'Small'
    else: 
        return 'Large'
    
titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1
titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type)

titanic_train['FamilySize'].describe()
sns.boxplot(x='FamilySize',data=titanic_train)
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.distplot, "FamilySize").add_legend()
sns.factorplot(x="Survived", hue="FamilyType", data=titanic_train, kind="count", size=6)

#process names of passengers
title_Dictionary = {
                        "Capt":       "Officer", "Col":        "Officer",
                        "Major":      "Officer", "Jonkheer":   "Royalty",
                        "Don":        "Royalty", "Sir" :       "Royalty",
                        "Dr":         "Officer", "Rev":        "Officer",
                        "the Countess":"Royalty","Dona":       "Royalty",
                        "Mme":        "Mrs", "Mlle":       "Miss",
                        "Ms":         "Mrs", "Mr" :        "Mr",
                        "Mrs" :       "Mrs", "Miss" :      "Miss",
                        "Master" :    "Master", "Lady" :      "Royalty"
}

def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

titanic_train['Title'] = titanic_train['Name'].map(extract_title)
titanic_train['Title'] = titanic_train['Title'].map(title_Dictionary)
    
pd.crosstab(index=titanic_train["Title"], columns="count")   
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Title'])
sns.factorplot(x="Survived", hue="Title", data=titanic_train, kind="count", size=6)

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.drop(['PassengerId','Name','Ticket','Cabin','Survived'], axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']

rf_grid = dict(n_estimators=list(range(100,1000,100)),
                      criterion=['gini','entropy'],
                      max_features=list(range(3,8,1)))
rf_estimator = ensemble.RandomForestClassifier(random_state=2017)
rf_grid_estimator = model_selection.GridSearchCV(estimator=rf_estimator, param_grid=rf_grid, cv=10, verbose=1, n_jobs=10, refit=True)
rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.grid_scores_
rf_grid_estimator.best_score_
rf_grid_estimator.best_params_
best_est = rf_grid_estimator.best_estimator_
best_est.feature_importances_

fi = pd.DataFrame({'features':titanic_train1.columns,
              'importance':best_est.feature_importances_})
#sns.factorplot(x="features", hue="importance", data=fi, kind="count", size=6)

gbm_tree_estimator = ensemble.GradientBoostingClassifier(n_estimators=5, random_state=2017)
gbm_grid = {'n_estimators':list(range(100,2000,100)),'learning_rate':[0.1,0.2, 0.3],'max_depth':list(range(3,9))}
gbm_grid_estimator = model_selection.GridSearchCV(gbm_tree_estimator,gbm_grid, cv=10, n_jobs=10)
gbm_grid_estimator.fit(X_train, y_train)
gbm_grid_estimator.grid_scores_
gbm_grid_estimator.best_score_
gbm_grid_estimator.best_params_

best_est = rf_grid_estimator.best_estimator_
best_est.feature_importances_