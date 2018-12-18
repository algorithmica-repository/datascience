import pandas as pd
import numpy as np
from sklearn import decomposition

X1 = [1,2,30,4,5]
X2 = [2,16,9,64,25]
X3 = [2,12,27,16,125]

df = pd.DataFrame({'X1':X1, 'X2':X2, 'X3':X3})
df.corr()

lpca = decomposition.PCA()
lpca.fit(df)
print(lpca.components_)
print(np.cumsum(lpca.explained_variance_ratio_))

lpca = decomposition.PCA(1)
df1 = lpca.fit_transform(df)
df2 = lpca.inverse_transform(df1)
