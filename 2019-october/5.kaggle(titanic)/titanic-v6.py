import pandas as pd
import os
from sklearn import tree, ensemble, model_selection, preprocessing
import io
import pydot
import seaborn as sns

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

age_imputer = preprocessing.Imputer()
age_imputer.fit(titanic_train[['Age']])
titanic_train['Age_imputed'] = age_imputer.transform(titanic_train[['Age']])

fare_imputer = preprocessing.Imputer()
fare_imputer.fit(titanic_train[['Fare']])

sns.countplot(x='Embarked',data=titanic_train)
titanic_train.loc[titanic_train['Embarked'].isnull(), 'Embarked'] = 'S'

sns.countplot(x='SibSp', data=titanic_train)
sns.distplot(titanic_train['SibSp'], hist=False)
sns.boxplot(x='SibSp',data=titanic_train)
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "SibSp").add_legend()

sns.countplot(x='Parch', data=titanic_train)
sns.distplot(titanic_train['Parch'], hist=False)
sns.boxplot(x='Parch',data=titanic_train)
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Parch").add_legend()

sex_encoder = preprocessing.LabelEncoder()
sex_encoder.fit(titanic_train['Sex'])
titanic_train['Sex_encoded'] = sex_encoder.transform(titanic_train['Sex'])

pclass_encoder = preprocessing.LabelEncoder()
pclass_encoder.fit(titanic_train['Pclass'])
titanic_train['Pclass_encoded'] = pclass_encoder.transform(titanic_train['Pclass'])

emb_encoder = preprocessing.LabelEncoder()
emb_encoder.fit(titanic_train['Embarked'])
titanic_train['Embarked_encoded'] = emb_encoder.transform(titanic_train['Embarked'])

#create title feature from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)
sns.factorplot(x="Title", hue="Survived", data=titanic_train, kind="count", size=6)

title_encoder = preprocessing.LabelEncoder()
title_encoder.fit(titanic_train['Title'])
titanic_train['Title_encoded'] = title_encoder.transform(titanic_train['Title'])

#create family size feature from sibsp, parch
titanic_train['FamilySize'] = titanic_train['SibSp'] +  titanic_train['Parch'] + 1
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic_train['FamilyGroup'] = titanic_train['FamilySize'].map(convert_familysize)
sns.factorplot(x="FamilyGroup", hue="Survived", data=titanic_train, kind="count", size=6)

fg_encoder = preprocessing.LabelEncoder()
fg_encoder.fit(titanic_train['FamilyGroup'])
titanic_train['FamilyGroup_encoded'] =fg_encoder.transform(titanic_train['FamilyGroup'])


features = ['SibSp', 'Parch', 'Fare', 'Pclass_encoded', 'Sex_encoded', 'Age_imputed', 'Embarked_encoded', 'Title_encoded', 'FamilySize', 'FamilyGroup_encoded']
X = titanic_train[ features ]
y = titanic_train['Survived']

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

rf_estimator = ensemble.RandomForestClassifier(random_state=100)
rf_grid = {'max_depth': [3,4,5,6], 'n_estimators':list(range(10, 50, 10)), 'max_features':[2,3,4]}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='accuracy', cv=10)
rf_grid_estimator.fit(X_train, y_train)
print(rf_grid_estimator.best_params_)
print(rf_grid_estimator.best_score_)
print(rf_grid_estimator.best_estimator_.estimators_)
print(rf_grid_estimator.score(X_train, y_train))

print(rf_grid_estimator.score(X_eval, y_eval))

base_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(base_estimator)
ada_grid = {'base_estimator__max_depth': [3,4,5], 'n_estimators':list(range(30, 200, 50)), 'learning_rate':[0.1,0.3,0.5,1.0]}
ada_grid_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, scoring='accuracy', cv=10)
ada_grid_estimator.fit(X_train, y_train)
print(ada_grid_estimator.best_params_)
print(ada_grid_estimator.best_score_)
print(ada_grid_estimator.best_estimator_.estimators_)
print(ada_grid_estimator.score(X_train, y_train))

print(ada_grid_estimator.score(X_eval, y_eval))

#grid search based model building
dt_estimator = tree.DecisionTreeClassifier()
dt_grid = {'max_depth': [3,4,5,6,7,8,9], 'criterion':['gini', 'entropy'], 'min_samples_split':[3, 5, 10]}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring='accuracy', cv=10)
dt_grid_estimator.fit(X_train, y_train)
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_estimator_)
print(dt_grid_estimator.score(X_train, y_train))

print(dt_grid_estimator.score(X_eval, y_eval))

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.info())

titanic_test['Age_imputed'] = age_imputer.transform(titanic_test[['Age']])
titanic_test['Fare'] = fare_imputer.transform(titanic_test[['Fare']])

titanic_test['FamilySize'] = titanic_test['SibSp'] +  titanic_test['Parch'] + 1
titanic_test['Title'] = titanic_test['Name'].map(extract_title)
titanic_test['FamilyGroup'] = titanic_test['FamilySize'].map(convert_familysize)
sns.countplot(x='Title',data=titanic_test)
sns.countplot(x='Title',data=titanic_train)
titanic_test.loc[titanic_test['Title']=='Dona', 'Title'] = 'Mrs'

titanic_test['Sex_encoded'] = sex_encoder.transform(titanic_test['Sex'])
titanic_test['Pclass_encoded'] = pclass_encoder.transform(titanic_test['Pclass'])
titanic_test['Embarked_encoded'] = emb_encoder.transform(titanic_test['Embarked'])
titanic_test['Title_encoded'] = title_encoder.transform(titanic_test['Title'])
titanic_test['FamilyGroup_encoded'] = fg_encoder.transform(titanic_test['FamilyGroup'])

X_test = titanic_test[features]
titanic_test['Survived'] = rf_grid_estimator.best_estimator_.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)

