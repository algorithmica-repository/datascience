import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np

print(pd.__version__)

class MyLabelBinarizer(preprocessing.LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((1-Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 1], threshold)
        else:
            return super().inverse_transform(Y, threshold)
        
path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

titanic_test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))
titanic_test.info()

def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)
#titanic_train['Title'] = titanic_train['Title'].astype('category')
print(titanic_train['Title'])

titanic_test['Title'] = titanic_test['Name'].map(extract_title)
#titanic_test['Title'] = titanic_test['Title'].astype('category')
print(titanic_test['Title'])

values = np.union1d(titanic_train['Title'], titanic_test['Title'])
list(values)
print(len(list(values)))
titanic_train['Title'] = titanic_train['Title'].astype(pd.api.types.CategoricalDtype(values))
titanic_test['Title'] = titanic_test['Title'].astype(pd.api.types.CategoricalDtype(values))



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

cat_features = ['Sex', 'Embarked', 'Pclass', 'Title']
cont_features = ['Age', 'Fare', 'SibSp', 'Parch']

feature_defs = []
for col_name in cat_features:
    feature_defs.append((col_name, MyLabelBinarizer()))
    
for col_name in cont_features:
    feature_defs.append((col_name, None))

mapper = DataFrameMapper(feature_defs, input_df=True, df_out=True)
mapper.fit(titanic_train)
titanic_train1 = mapper.transform(titanic_train)
print(mapper.transformed_names_)

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test1 = mapper.transform(titanic_test)
