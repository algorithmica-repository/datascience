import pandas as pd
import os
from sklearn import model_selection, ensemble, naive_bayes, neighbors, linear_model
import mlxtend.classifier as mlx

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

features = ['Sex', 'Pclass', 'Embarked','Parch','SibSp']
titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Survived','Name','Age','Cabin','Ticket'], axis=1)
y_train = titanic_train['Survived']

knn = neighbors.KNeighborsClassifier()
gnb = naive_bayes.GaussianNB()
rf = ensemble.RandomForestClassifier(random_state=100)
lr = linear_model.LogisticRegression(random_state=100)


sclf = mlx.StackingClassifier(classifiers=[knn, gnb, rf], 
                          meta_classifier=lr, 
                          store_train_meta_features=True,
                          use_probas=True)
st_grid = {'kneighborsclassifier__n_neighbors': [3,4,5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta-logisticregression__C': [0.1, 10.0]}

grid_classifier = model_selection.GridSearchCV(sclf, st_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_
meta = final_model.train_meta_features_

titanic_test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))
print(titanic_test.shape)
print(titanic_test.info())
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Sex','Pclass','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Name','Age','Cabin','Ticket'], axis=1)
titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)

