import pandas as pd
from sklearn import tree
import pydot
import io
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

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
y_train = titanic_train['Survived']
#create an instance of decision tree classifier type
dt_estimator = tree.DecisionTreeClassifier()

#grid search 
dt_grid = {'criterion':["gini", "entropy"], 'max_depth':[3,4,5,6,7], 'min_samples_split':[2,10,20,30]}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
dt_grid_estimator.fit(X_train, y_train)

#random search
dt_grid_estimator = model_selection.RandomizedSearchCV(dt_estimator, dt_grid, cv=10, n_iter=20)
dt_grid_estimator.fit(X_train, y_train)

#access the results
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
final_estimator = dt_grid_estimator.best_estimator_
results = dt_grid_estimator.cv_results_
print(results.get("mean_test_score"))
print(results.get("mean_train_score"))
print(results.get("params"))


#get the logic or model learned by Algorithm
#issue: not readable
print(final_estimator.tree_)

#get the readable tree structure from tree_ object
#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(final_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("C:/Users/Algorithmica/Downloads/tree.pdf")

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
