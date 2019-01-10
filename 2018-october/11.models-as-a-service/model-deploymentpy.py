from sklearn.externals import joblib
import pandas as pd
from sklearn import tree, model_selection, preprocessing
from sklearn_pandas import CategoricalImputer
import numpy as np
import seaborn as sns
import os

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

#impute missing values for continuous features
imputable_cont_features = ['Age', 'Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

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

#label encoding of categorical (string) features
lab_encoder_sex = preprocessing.LabelEncoder()
lab_encoder_sex.fit(titanic_train['Sex'])
print(lab_encoder_sex.classes_)
titanic_train['Sex'] = lab_encoder_sex.transform(titanic_train['Sex'])

lab_encoder_emb = preprocessing.LabelEncoder()
lab_encoder_emb.fit(titanic_train['Embarked'])
print(lab_encoder_emb.classes_)
titanic_train['Embarked'] = lab_encoder_emb.transform(titanic_train['Embarked'])

lab_encoder_pclass = preprocessing.LabelEncoder()
lab_encoder_pclass.fit(titanic_train['Pclass'])
print(lab_encoder_pclass.classes_)
titanic_train['Pclass'] = lab_encoder_pclass.transform(titanic_train['Pclass'])

lab_encoder_fgroup = preprocessing.LabelEncoder()
lab_encoder_fgroup.fit(titanic_train['FamilyGroup'])
print(lab_encoder_fgroup.classes_)
titanic_train['FamilyGroup'] = lab_encoder_fgroup.transform(titanic_train['FamilyGroup'])

#one hot encoding of categorical integer features
ohe_features = ['Sex','Embarked','Pclass', 'FamilyGroup']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic_train[ohe_features])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic_train[ohe_features]).toarray()

features = ['Age', 'Fare', 'Parch' , 'SibSp', 'FamilySize']
tmp2 = titanic_train[features].values

X_train = np.concatenate((tmp1,tmp2), axis=1)
y_train = titanic_train['Survived']

#create an estimator 
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':[3,4,5,6,7], 'criterion':['entropy','gini'] }
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring='accuracy', cv=10, refit=True)
dt_grid_estimator.fit(X_train, y_train)

#explore the results of grid_search_cv estimator
print(dt_grid_estimator.cv_results_)
print(dt_grid_estimator.best_estimator_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_params_)

best_dt_estimator = dt_grid_estimator.best_estimator_
print(best_dt_estimator.score(X_train, y_train))


objects_to_dump = { 
        'imputable-features': imputable_cont_features,
        'cont-features': features,
        'cat-features':ohe_features,
        'cont-imputer':cont_imputer,
        'cat-imputer':cat_imputer,
        'family-size-func':convert_familysize,
        'le-sex':lab_encoder_sex,
        'le-emb':lab_encoder_emb,
        'le-pclass':lab_encoder_pclass,
        'le-fgroup':lab_encoder_fgroup,
        'ohe':ohe,
        'estimator':best_dt_estimator
        }
path = 'C:\\Users\\Algorithmica\\Downloads'
joblib.dump(objects_to_dump, os.path.join(path, 'deployment.pkl') )

objects_loaded = joblib.load(os.path.join(path, 'deployment.pkl') )
print(objects_loaded.get('cont-features'))
print(objects_loaded.get('le-sex'))

