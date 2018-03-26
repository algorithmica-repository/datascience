import pandas as pd
#build data frame from dictionary
map1 = {'id':[1,2,3], 'fare':[13, 15.7, 25]}
df = pd.DataFrame(map1)
type(df)
df.shape
df.info()

df['fare']
df.iloc[0:2,:]
