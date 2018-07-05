#2-d container with heterogeneous columns
import pandas as pd

#read csv file into dataframe 
titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(type(titanic_train))
print(titanic_train.shape)

#understanding shape information
s = titanic_train.shape
print(s[0])
print(s[1])

#retrieve columns
titanic_train['Pclass']
titanic_train.Pclass
tmp1 = titanic_train[['Fare']]
print(type(tmp1))

tmp2 = titanic_train['Fare']
print(type(tmp2))

columns = ['Pclass','Fare']
tmp = titanic_train[columns]
print(type(tmp))

#retrieve rows
titanic_train[0:2]
titanic_train[10:16]

#retrieve subset of rows & columns
#for indexed based access, use iloc
#for named access, use loc
titanic_train.iloc[0:3, 0:2]
titanic_train.iloc[10:17, 4:8]
titanic_train.loc[0:3, columns]

#conditional access
titanic_train[titanic_train['Pclass']==3]
titanic_train[titanic_train.Survived == 1]
titanic_train[(titanic_train.Survived == 1) & (titanic_train.Sex == 'male')]

#add new columns
titanic_train['dummy'] = 1
