import pandas as pd
import numpy as np
from sklearn import decomposition

X1 = [1,2,3,4,5]
X2 = [2,4,6,8,10]
X3 = [5,4,3,2,1]

df = pd.DataFrame({'X1':X1, 'X2':X2, 'X3':X3})
df.corr()

lpca = decomposition.PCA()
lpca.fit(df)
print(lpca.components_)
print(np.cumsum(lpca.explained_variance_ratio_))

lpca = decomposition.PCA(1)
df1 = lpca.fit_transform(df)
df2 = lpca.inverse_transform(df1)
