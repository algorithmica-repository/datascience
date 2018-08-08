import pandas as pd
from sklearn import neighbors, model_selection, preprocessing
from sklearn_pandas import CategoricalImputer
import pandas
print(pandas.__version__)

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

titanic_train['Sex'] = titanic_train['Sex'].astype('category', categories=["male","female"], ordered=False)
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category', categories=[1, 2, 3], ordered=False)
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category', categories=["C","Q","S"], ordered=False)


#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

cat_columns = ['Sex', 'Embarked', 'Pclass']
titanic_train1 = pd.get_dummies(titanic_train, columns = cat_columns)
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

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])

titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
titanic_test1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test = scaler.transform(titanic_test1)
titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)

