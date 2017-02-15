import pandas as pd

col1 = [10,20,30,40]
col2 = ['abc','def','xyz','pqr']
col3 = [0,0,0,0]

#creating data frame
df1 = pd.DataFrame({'pid':col1,
'pname':col2,'survived':col3})
df1.shape
df1.info()
df1.describe()
df1.head(2)
df1.tail()

df1['col4'] = 0

#access frame content by column/columns
df1.pid
df1['pid']
df1[['pid','pname']]
df1[[0,1]]

#dropping a column
df2 = df1.drop('survived',1)

#slicing rows of frame
df1[0:2]
df1[0:4]
df1[0:]
df1[:2]
df1[-2:]

#filtering rows of dataframe by condition
type(df1.pid > 20)
df1[df1.pid>20]

#selecting subsets of rows and columns
df1.iloc[0:2,]
df1.iloc[[0,2],]
df1.iloc[0:2,0]
df1.iloc[0:2,[0,2]]
df1.loc[0:2,['pname']]

#grouping data in data frames
df1.groupby('id').size()
