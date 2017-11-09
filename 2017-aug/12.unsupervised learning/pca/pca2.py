import numpy as np
from sklearn import decomposition
import pandas as pd

df1= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[20,4,16,18,24],
        'F3':[30,6,24,27,36]})
pca = decomposition.PCA()
pca.fit(df1)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

df2= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[-120,9,10,60,100]})
np.corrcoef(df2.F1, df2.F2)
pca = decomposition.PCA() 
pca.fit(df2)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
