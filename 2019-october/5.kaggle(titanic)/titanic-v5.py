import pandas as pd
import os
from sklearn import tree, ensemble, model_selection, preprocessing
import io
import pydot

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

lencoder = preprocessing.LabelEncoder()
lencoder.fit(titanic_train['Sex'])
print(lencoder.classes_)
titanic_train['Sex_encoded'] = lencoder.transform(titanic_train['Sex'])

imputer = preprocessing.Imputer()
imputer.fit(titanic_train[['Age']])
print(imputer.statistics_)
titanic_train['Age_imputed'] = imputer.transform(titanic_train[['Age']])

features = ['SibSp', 'Parch', 'Pclass', 'Sex_encoded', 'Age_imputed']
X = titanic_train[ features ]
y = titanic_train['Survived']

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

#grid search based model building
rf_estimator = ensemble.RandomForestClassifier(random_state=100)
rf_grid = {'max_depth': [3,4,5,6], 'n_estimators':list(range(10, 50, 10)), 'max_features':[2,3,4]}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, scoring='accuracy', cv=10)
rf_grid_estimator.fit(X_train, y_train)
print(rf_grid_estimator.best_params_)
print(rf_grid_estimator.best_score_)
print(rf_grid_estimator.best_estimator_.estimators_)
print(rf_grid_estimator.score(X_train, y_train))

print(rf_grid_estimator.score(X_eval, y_eval))

#visualize the deciion tree
for i, est in enumerate(rf_grid_estimator.best_estimator_.estimators_):
    X_df = pd.DataFrame(X_train, columns=X_train.columns)
    dot_data = io.StringIO() 
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_df.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(os.path.join(dir, "tree" + str(i) + ".pdf"))
