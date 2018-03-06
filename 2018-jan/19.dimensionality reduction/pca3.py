from sklearn import decomposition
import pandas as pd

df1= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[1,1,1,1,0],
        'F3':[1,9,2,1,0]})
pca = decomposition.PCA()
pca.fit(df1)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#reduce dimensions to 1 after observing the ratio of variances along pcs
pca = decomposition.PCA(1)
pca.fit(df1)
print(pca.explained_variance_)

#tranform original data to new pc dimensions
df2 = pca.transform(df1)