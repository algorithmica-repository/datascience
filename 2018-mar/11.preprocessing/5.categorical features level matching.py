import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np
  
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
titanic_train['Title'] = titanic_train['Title'].astype(pd.api.types.CategoricalDtype(categories=values))
titanic_test['Title'] = titanic_test['Title'].astype(pd.api.types.CategoricalDtype(categories=values))



