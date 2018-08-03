import pandas as pd
from sklearn import neighbors, model_selection, preprocessing

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

titanic_train.loc[titanic_train['Age'].isnull() == True, 'Age'] = titanic_train['Age'].mean()

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

titanic_test.loc[titanic_test['Age'].isnull() == True, 'Age'] = titanic_test['Age'].mean()
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()
titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
titanic_test1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test = scaler.transform(titanic_test1)
titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)

