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
age_imputer = preprocessing.Imputer()
#it computes the information required for imputation algorithm
age_imputer.fit(titanic_train[['Age']])
#It applies the imputation algorithm on complete feature and returns transformed data
titanic_train[['Age']] = age_imputer.transform(titanic_train[['Age']])

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
sns.distplot(titanic_train['FamilySize'])
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

#process ticket feature
def extract_id(ticket):        
        id = ticket.replace('.','').replace('/','').split()[0]
        if not id.isdigit() and len(id) > 0:
            return id.upper()
        else: 
            return 'X'

titanic_train['TicketId'] = titanic_train['Ticket'].map(extract_id)

pd.crosstab(index=titanic_train["TicketId"], columns="count")   
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['TicketId'])
sns.factorplot(x="Survived", hue="TicketId", data=titanic_train, kind="count", size=6)

#process fare column
fare_imputer = preprocessing.Imputer(strategy="median")
fare_imputer.fit(titanic_train[['Fare']])

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title','TicketId'])
type(titanic_train1)
titanic_train1.info()
titanic_train1.drop(['PassengerId','Name','FamilySize','SibSp','Parch','Ticket','Cabin','Survived'], axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']

parameter_grid = dict(n_estimators=[300,400],
                      criterion=['gini','entropy'],
                      max_features=[3,4,5,6,7,8])
rf_estimator = ensemble.RandomForestClassifier(random_state=100)
rf_grid_estimator = model_selection.GridSearchCV(estimator=rf_estimator, param_grid=parameter_grid, cv=10, verbose=1, n_jobs=10, refit=True)
rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.grid_scores_

titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.info()

#explore missing data
titanic_test.apply(lambda x : sum(x.isnull()))

#pre-process Age
titanic_test[['Age']] = age_imputer.transform(titanic_test[['Age']])

#pre-process Fare
titanic_test[['Fare']] = fare_imputer.transform(titanic_test[['Fare']])

titanic_test['FamilySize'] = titanic_test.SibSp + titanic_test.Parch + 1
titanic_test['FamilyType'] = titanic_test['FamilySize'].map(size_to_type)

titanic_test['Title'] = titanic_test['Name'].map(extract_title)
titanic_test['Title'] = titanic_test['Title'].map(title_Dictionary)
    
titanic_test['TicketId'] = titanic_test['Ticket'].map(extract_id)

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title','TicketId'])
titanic_test1.info()

titanic_test1.drop(['PassengerId','Name','FamilySize','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)

X_test = titanic_test1

#number of features in test data mismatches with train data
titanic_test1['Survived'] = rf_grid_estimator.predict(X_test)
