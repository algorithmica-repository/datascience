import pandas as pd
from sklearn import tree, model_selection, preprocessing
import pydot
import io
from sklearn_pandas import CategoricalImputer
import numpy as np

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

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

#label encoding of categorical (string) features
lab_encoder_sex = preprocessing.LabelEncoder()
lab_encoder_sex.fit(titanic_train['Sex'])
print(lab_encoder_sex.classes_)
titanic_train['Sex'] = lab_encoder_sex.transform(titanic_train['Sex'])

lab_encoder_emb = preprocessing.LabelEncoder()
lab_encoder_emb.fit(titanic_train['Embarked'])
print(lab_encoder_emb.classes_)
titanic_train['Embarked'] = lab_encoder_emb.transform(titanic_train['Embarked'])

lab_encoder_pclass = preprocessing.LabelEncoder()
lab_encoder_pclass.fit(titanic_train['Pclass'])
print(lab_encoder_pclass.classes_)
titanic_train['Pclass'] = lab_encoder_pclass.transform(titanic_train['Pclass'])

#one hot encoding of categorical integer features
ohe_features = ['Sex','Embarked','Pclass']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic_train[ohe_features])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic_train[ohe_features]).toarray()

features = ['Age', 'Fare', 'Parch' , 'SibSp']
tmp2 = titanic_train[features].values

X_train = np.concatenate((tmp1,tmp2), axis=1)
y_train = titanic_train['Survived']

#create an estimator 
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':[3,4,5,6,7], 'criterion':['entropy','gini'] }
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring='accuracy', cv=10, refit=True)
dt_grid_estimator.fit(X_train, y_train)

#explore the results of grid_search_cv estimator
print(dt_grid_estimator.cv_results_)
print(dt_grid_estimator.best_estimator_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_params_)

#visualuze the final model built with best parameters in grid
best_dt_estimator = dt_grid_estimator.best_estimator_
print(best_dt_estimator.score(X_train, y_train))

print(best_dt_estimator.tree_)
dot_data = io.StringIO() 
tree.export_graphviz(best_dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("C:/Users/Algorithmica/Downloads/tree2.pdf")

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Sex'] = lab_encoder_sex.transform(titanic_test['Sex'])
titanic_test['Embarked'] = lab_encoder_emb.transform(titanic_test['Embarked'])
titanic_test['Pclass'] = lab_encoder_pclass.transform(titanic_test['Pclass'])
tmp1 = ohe.transform(titanic_test[ohe_features]).toarray()
tmp2 = titanic_test[features].values

X_test = np.concatenate((tmp1,tmp2), axis=1)
titanic_test['Survived'] = best_dt_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)

