import pandas as pd
from sklearn import neighbors, model_selection, preprocessing
from sklearn_pandas import CategoricalImputer

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
print(titanic_test.info())

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])

titanic = pd.concat([titanic_train, titanic_test])
titanic1 =  pd.get_dummies(titanic, columns= ['Sex', 'Embarked', 'Pclass'] )

titanic_train1 = titanic1[:titanic_train.shape[0]]
print(titanic_train1.shape)

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train1[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train1[imputable_cont_features] = cont_imputer.transform(titanic_train1[imputable_cont_features])

#Feature filteration
titanic_train1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
scaler.fit(titanic_train1)
print(scaler.mean_)
print(scaler.var_)
X_train = scaler.transform(titanic_train1)
y_train = titanic_train['Survived']

classifier = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':list(range(1,12,1)), 'weights':['uniform','distance']}
grid_classifier = model_selection.GridSearchCV(classifier, knn_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_

titanic_test = titanic1[titanic_train.shape[0]:]
print(titanic_test.shape)
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])

titanic_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test = scaler.transform(titanic_test)
titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)

