import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np

#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")
#os.chdir("/home/algo/Downloads")


titanic_train = pd.read_csv("titanic_train.csv")

#EDA
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

encodable_columns=['Sex', 'Embarked', 'Pclass']
feature_defs = [(col_name, preprocessing.LabelEncoder()) for col_name in encodable_columns]
mapper = DataFrameMapper(feature_defs)
mapper.fit(titanic_train)
titanic_train[encodable_columns] = mapper.transform(titanic_train)

titanic_train.drop(['PassengerId', 'Name', 'Cabin','Ticket'], axis=1, inplace=True)

one_hot_encoder = preprocessing.OneHotEncoder(categorical_features = np.array([1,2,7]))
one_hot_encoder.fit(titanic_train)
print(one_hot_encoder.n_values_)
titanic_train1 = one_hot_encoder.transform(titanic_train).toarray()
