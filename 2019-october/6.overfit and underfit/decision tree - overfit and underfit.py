import pandas as pd
import os
from sklearn import tree, model_selection, preprocessing
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
X_train = titanic_train[ features ]
y_train = titanic_train['Survived']
dt_estimator = tree.DecisionTreeClassifier(max_depth=1, min_samples_split=200)
dt_estimator.fit(X_train, y_train)
print(dt_estimator.tree_)
print(model_selection.cross_val_score(dt_estimator, X_train, y_train, scoring="accuracy", cv=10).mean())
print(dt_estimator.score(X_train, y_train))


#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
dir = 'E:/'
graph.write_pdf(os.path.join(dir, "tree2.pdf"))

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.info())

#fit must only be performed only on train data since it must reflect the learnign from train data only
titanic_test['Sex_encoded'] = lencoder.transform(titanic_test['Sex'])
titanic_test['Age_imputed'] = imputer.transform(titanic_test[['Age']])

X_test = titanic_test[features]
titanic_test['Survived'] = dt_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
