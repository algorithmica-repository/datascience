import pandas as pd

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/titanic_train.csv")
#get size of the frame
print(titanic_train.shape)
shape = titanic_train.shape
print(shape[0])
print(shape[1])

#get types of each column
print(titanic_train.info())

#get rows of frame
titanic_train[2:11]
titanic_train[800:]
titanic_train[:10]

#get columns of frame
print(titanic_train['Survived'])
print(titanic_train['Pclass'])
print(titanic_train[['Pclass','Survived']])
print(titanic_train.Pclass)

#access rows and columns
titanic_train.iloc[:3, 4:8]
titanic_train.loc[:3, 'Survived']
titanic_train.loc[:3, ['Survived', 'Pclass'] ]


#conditional access
titanic_train.loc[titanic_train.Sex=='female', ['Sex','Survived']]

#aggregates on frame
titanic_train.groupby('Survived').size()
titanic_train.groupby(['Pclass','Survived']).size()
titanic_train.groupby(['Sex','Survived']).size()
titanic_train.groupby(['Age','Survived']).size()
titanic_train.groupby(['Sex','Pclass','Survived']).size()
