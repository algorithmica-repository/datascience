import numpy as np
from sklearn import decomposition
import pandas as pd

df1= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[3,5,0.12,3,2],
        'F3':[0.1,3,7,6,24]})
print(df1.corr())

pca1 = decomposition.PCA()

#learn the new principal axis
pca1.fit(df1)
#transform original data to new axis
df2 = pca1.transform(df1)

#variance of data along original dimensions = variance of data along new axis
tot_var_original = np.trunc(np.var(df1.F1) + np.var(df1.F2) + np.var(df1.F3))
tot_var_transformed = np.trunc(np.var(df2[:,0]) + np.var(df2[:,1]) + np.var(df2[:,2]))

#principal components captures variance in decreasing order
print(pca1.explained_variance_ratio_)

pca2 = decomposition.PCA(n_components=1)
pca2.fit(df1)
df3 = pca2.transform(df1)
print(np.var(df3[:,0]))
