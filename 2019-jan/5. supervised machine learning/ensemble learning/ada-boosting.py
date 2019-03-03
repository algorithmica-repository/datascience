import pandas as pd
from sklearn import ensemble, tree, neighbors
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer

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

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
y_train = titanic_train['Survived']

dt_estimator = tree.DecisionTreeClassifier()
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=dt_estimator)
ada_grid = {'n_estimators':[10, 50, 100, 200], 'base_estimator__max_depth':[3,4,5,6,7], 'learning_rate':[0.1, 0.5, 1] }
ada_grid_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, cv=10, return_train_score=True)
ada_grid_estimator.fit(X_train, y_train)

print(ada_grid_estimator.best_score_)
print(ada_grid_estimator.best_params_)
final_estimator = ada_grid_estimator.best_estimator_
final_estimator.score(X_train, y_train)
print(final_estimator.estimators_)

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived'] = final_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)
