import os
import pandas as pd
from sklearn import preprocessing

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

scalable_features = ['Age','Fare']
scaler1 = preprocessing.StandardScaler()
scaler1.fit(titanic_train[scalable_features])
print(scaler1.scale_)
print(scaler1.var_)
titanic_train[scalable_features] = scaler1.transform(titanic_train[scalable_features])


# =============================================================================
# scaler2 = preprocessing.MinMaxScaler()
# scaler2.fit(titanic_train[scalable_features])
# print(scaler2.data_min_)
# print(scaler2.data_max_)
# titanic_train[scalable_features] = scaler2.transform(titanic_train[scalable_features])
# 
# scaler3 = preprocessing.RobustScaler()
# scaler3.fit(titanic_train[scalable_features])
# print(scaler3.center_)
# print(scaler3.scale_)
# titanic_train[scalable_features] = scaler3.transform(titanic_train[scalable_features])
# 
# =============================================================================
