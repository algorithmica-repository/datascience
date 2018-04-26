import pandas as pd
import os
from sklearn import tree, ensemble, model_selection
import io
import pydot

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

features = ['Sex', 'Pclass', 'Embarked','Parch','SibSp']
titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Survived','Name','Age','Cabin','Ticket'], axis=1)
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier(random_state=100)
classifier = ensemble.AdaBoostClassifier(dt, random_state=100, algorithm='SAMME')
ada_grid = {'base_estimator__max_depth':[3,4,5], 'n_estimators':[10], 'learning_rate':[0.1,1.0]}
grid_classifier = model_selection.GridSearchCV(classifier, ada_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
final_model = grid_classifier.best_estimator_
print(final_model.estimator_weights_)
print(final_model.estimator_errors_)
print(final_model.estimators_)

#extracting all the trees build by random forest algorithm
n_tree = 0
for est in final_model.estimators_: 
    dot_data = io.StringIO()
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf("ada" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1


titanic_test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))
print(titanic_test.shape)
print(titanic_test.info())
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Sex','Pclass','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Name','Age','Cabin','Ticket'], axis=1)
titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
