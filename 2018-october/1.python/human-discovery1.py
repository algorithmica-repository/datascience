import pandas as pd

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(type(titanic_train))
print(titanic_train.info())

#analyze survived column for majority level

#groupby
pd.groupby(titanic_train, ['Survived']).size()

#crosstab