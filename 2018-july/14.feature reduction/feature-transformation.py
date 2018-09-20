import numpy as np
from sklearn import decomposition
import pandas as pd

df1= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[20,5,17,20,22],
        'F3':[10,2,7,10,11]})
print(df1.corr())

pca = decomposition.PCA()

#learn the new principal axis
pca.fit(df1)
#transform original data to new axis
df2 = pca.transform(df1)

#variance of data along original dimensions = variance of data along new axis
tot_var_original = np.trunc(np.var(df1.F1) + np.var(df1.F2) + np.var(df1.F3))
tot_var_transformed = np.trunc(np.var(df2[:,0]) + np.var(df2[:,1]) + np.var(df2[:,2]))

assert tot_var_original == tot_var_transformed

#variance of data along principal component axes
tot_var_pca = np.trunc(np.sum(pca.explained_variance_))

#principal components captures variance in decreasing order
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#transform original data to PC dimensions
df2 = pca.transform(df1)

#build pca model for given data
#principal components are just original dimensions rotated by some angle
#relation between given features to principal components
#PC1 = w11*F1 + w12*F2 + w13*F3
#PC2 = w21*F1 + w22*F2 + w23*F3
#PC3 = w31*F1 + w32*F2 + w33*F3