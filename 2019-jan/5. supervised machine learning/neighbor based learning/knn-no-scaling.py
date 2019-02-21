import pandas as pd
from sklearn import neighbors
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

knn_estimator = neighbors.KNeighborsClassifier()
knn_estimator.fit(X_train, y_train)

scores = model_selection.cross_validate(knn_estimator, X_train, y_train, cv = 10)
test_scores = scores.get("test_score")
print(test_scores.mean())

train_scores = scores.get("train_score")
print(train_scores.mean())


#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived'] = knn_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)
