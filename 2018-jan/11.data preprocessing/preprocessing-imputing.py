import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import CategoricalImputer

#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

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
