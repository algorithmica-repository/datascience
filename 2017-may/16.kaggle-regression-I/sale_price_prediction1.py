import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/regression")
#os.chdir("/home/algo/Downloads")

house_train = pd.read_csv("house_train.csv")

#EDA
house_train.shape
house_train.info()

#extract numerical and continuous columns of data frame
numeric_cols = house_train.select_dtypes(include=['number']).columns
cat_cols = house_train.select_dtypes(exclude = ['number']).columns

#convert numerical columns to categorical type              
house_train['MSSubClass'] = house_train['MSSubClass'].astype('category')

#convert categorical columns to numeric type
ordinal_features1 = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
#ordinal_features1 = [col for col in house_train if 'TA' in list(house_train[col])]
quality_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for feature in ordinal_features1:
    null_idx = house_train[feature].isnull()
    house_train.loc[null_idx, feature] = None 
    house_train[feature] = house_train[feature].map(quality_dict)


house_train.describe()
#smooth the sale price using log transformation(smoothening outlier data)
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
sns.distplot(house_train['SalePrice'],kde=False)
sns.distplot(house_train['log_sale_price'],kde=False)


house_train.drop(["Id"],axis=1,inplace=True)
#handle missing data columns
total_missing = house_train.isnull().sum()
type(total_missing)
to_delete = total_missing[total_missing>0]
house_train.drop(list(to_delete.index),axis=1,inplace=True)
house_train.info()

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = house_train)
plt.xticks(rotation=45)

sns.boxplot(x = 'OverallQual', y = 'SalePrice',  data = house_train)
sns.boxplot(x = 'CentralAir', y = 'SalePrice',  data = house_train)
sns.boxplot(x = 'YearBuilt', y = 'SalePrice',  data = house_train)
plt.xticks(rotation=45)

sns.jointplot(x = "GrLivArea", y = "SalePrice", data = house_train)
sns.jointplot(x = "TotalBsmtSF", y = "SalePrice", data = house_train)

corr = house_train.select_dtypes(include = ['number']).corr()
type(corr)
sns.heatmap(corr, square=True)
plt.xticks(rotation=70)
plt.yticks(rotation=70)
corr_list1 = corr['SalePrice'].sort_values(axis=0,ascending=False)
corr_list2 = corr['log_sale_price'].sort_values(axis=0,ascending=False)

numeric_cols = house_train.select_dtypes(include=['number']).columns
cat_cols = house_train.select_dtypes(exclude = ['number']).columns

house_train.shape
house_train1 = pd.get_dummies(house_train, columns=cat_cols)
house_train1.shape

X_train = house_train1.drop(['SalePrice','log_sale_price'],axis=1)
y_train = house_train['log_sale_price']
dt_estimator = tree.DecisionTreeRegressor(random_state=2017)
model_selection.cross_val_score(dt_estimator, X_train, y_train, cv=10,scoring="r2").mean()
dt_estimator.fit(X_train, y_train)


house_test = pd.read_csv("house_test.csv")
house_test.shape
house_test.info()

house_test['MSSubClass'] = house_test['MSSubClass'].astype('category')

for feature in ordinal_features1:
    null_idx = house_test[feature].isnull()
    house_test.loc[null_idx, feature] = None 
    house_test[feature] = house_test[feature].map(quality_dict)

house_test1 = house_test.drop(["Id"],axis=1)

house_test1.drop(list(to_delete.index),axis=1,inplace=True)
house_test1.shape
house_test2 = pd.get_dummies(house_test1, columns=cat_cols)
house_test2.shape

X_test = house_test2
house_test['log_sales_price'] = dt_estimator.predict(X_test)
#prediction fails because of mismatch in number of features