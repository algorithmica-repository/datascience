import pandas as pd

#create custom frames with dictionary

df1 = pd.DataFrame({'age':[10,20,30],
                    'pclass':[1,2,3]})
print(type(df1))
print(df1)

#create frames from readers
df2 = pd.read_csv()

df3 = pd.read_json()

df4 = pd.read_table()

#combine data frames

#sort frame by column/feature



