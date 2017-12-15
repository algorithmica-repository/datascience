import pandas as pd

#create custom dataframe
passengers = { 'passenger_id': [10,11,12], 'fare':[123.4,12.6,30] }
df1 = pd.DataFrame(passengers)
df1.shape
df1.info()

print(df1['passenger_id'])
print(df1['fare'])

print(df1.loc[2,'fare'])

#adding new column to existing data frame
df1['pclass'] = 1
