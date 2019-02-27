import pandas as pd
from sklearn import neighbors
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer
from sklearn.externals import joblib
import os

dir = "C:\\Users\\Algorithmica\\Downloads"
#creation of data frames from csv
titanic_train = pd.read_csv(os.path.join(dir, "titanic_train.csv"))
print(titanic_train.info())

#preprocessing stage
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

le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Embarked'])
print(le_embarked.classes_)
titanic_train['Embarked'] = le_embarked.transform(titanic_train['Embarked'])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(titanic_train['Sex'])
print(le_sex.classes_)
titanic_train['Sex'] = le_sex.transform(titanic_train['Sex'])

le_pclass = preprocessing.LabelEncoder()
le_pclass.fit(titanic_train['Pclass'])
print(le_pclass.classes_)
titanic_train['Pclass'] = le_pclass.transform(titanic_train['Pclass'])

categorical_features = ['Pclass', 'Sex', 'Embarked']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic_train[categorical_features])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic_train[categorical_features]).toarray()
tmp1 = pd.DataFrame(tmp1)

continuous_features = ['Fare', 'Age', 'SibSp', 'Parch']
tmp2 = titanic_train[continuous_features]

tmp = pd.concat([tmp1, tmp2], axis=1)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(tmp)
y_train = titanic_train['Survived']

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':[5,7,8,10,20], 'weights':['uniform','distance']}
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, cv=10, return_train_score=True)
knn_grid_estimator.fit(X_train, y_train)

print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.best_params_)
results = knn_grid_estimator.cv_results_
final_estimator = knn_grid_estimator.best_estimator_
print(final_estimator.score(X_train, y_train))

objects_to_dump = { 
        'imputable_cont_features': imputable_cont_features,
        'cont_imputer':cont_imputer,
        'cat_imputer':cat_imputer,
        'le_sex':le_sex,
        'le_emb':le_embarked,
        'le_pclass':le_pclass,
        'ohe':ohe,
        'knn_estimator':final_estimator,
        'cont_features': continuous_features,
        'cat_features':categorical_features,
        'scaler': scaler
      }
joblib.dump(objects_to_dump, os.path.join(dir, 'deployment.pkl') )
