import pandas as pd

df = pd.DataFrame({'id':[1,2,3], 'fare':[13, 15.7, 25]})
type(df)
df.shape
df.info()

df['fare']
df.iloc[0:2]