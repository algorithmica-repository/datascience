#load the module to current session
import pandas as pd

titanic_train = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_train.csv')

#column level access
print(titanic_train['Survived'])
print(titanic_train[['Survived','Sex']])

#row level access
print(titanic_train[0:3])
print(titanic_train[10:])
print(titanic_train[40:61])

#row and column access
print(titanic_train.iloc[0:3,1:3])
