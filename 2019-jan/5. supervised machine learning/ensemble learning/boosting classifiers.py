import pandas as pd
from sklearn import neighbors
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer
import xgboost as xgb

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
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


#ada boost
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),random_state=100)
ada_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'base_estimator__max_depth':[1,3,5], 'base_estimator__criterion':['entropy', 'gini'] }
ada_grid_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, scoring='accuracy', cv=10, return_train_score=True)
ada_grid_estimator.fit(X_train, y_train)

print(ada_grid_estimator.best_score_)
print(ada_grid_estimator.best_params_)
final_estimator = ada_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.estimator_weights_)
print(final_estimator.estimator_errors_)
print(final_estimator.score(X_train, y_train))

#gradient boosting
gb_estimator = ensemble.GradientBoostingClassifier(random_state=100)
gb_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5]}
gb_grid_estimator = model_selection.GridSearchCV(gb_estimator, gb_grid, scoring='accuracy', cv=10, return_train_score=True)
gb_grid_estimator.fit(X_train, y_train)

print(gb_grid_estimator.best_score_)
print(gb_grid_estimator.best_params_)
final_estimator = gb_grid_estimator.best_estimator_
print(final_estimator.estimators_)
print(final_estimator.score(X_train, y_train))

#xgb = boosting + regulaization for overfit control
xgb_estimator = xgb.XGBClassifier(random_state=100, n_jobs=-1)
xgb_grid = {'n_estimators':list(range(50,101,50)), 'learning_rate':[0.1,0.2,1.0], 'max_depth':[1,3,5], 'gamma':[0,0.01,0.1,0.2], 'reg_alpha':[0,0.5,1], 'reg_lambda':[0,0.5,1]}
xgb_grid_estimator = model_selection.GridSearchCV(xgb_estimator, xgb_grid, scoring='accuracy', cv=10, return_train_score=True)
xgb_grid_estimator.fit(X_train, y_train)

print(xgb_grid_estimator.best_score_)
print(xgb_grid_estimator.best_params_)
final_estimator = xgb_grid_estimator.best_estimator_
print(final_estimator.score(X_train, y_train))

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])
titanic_test['Pclass'] = le_pclass.transform(titanic_test['Pclass'])
tmp1 = ohe.transform(titanic_test[categorical_features]).toarray()
tmp1 = pd.DataFrame(tmp1)
tmp2 = titanic_test[continuous_features]
tmp = pd.concat([tmp1, tmp2], axis=1)

X_test = scaler.transform(tmp)
titanic_test['Survived'] = final_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)