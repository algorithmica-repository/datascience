from sklearn import decomposition
import seaborn as sns
import pandas as pd

#create dataframe with 100% correlation
df1 = pd.DataFrame({'x1':[10,20,30,40],'x2':[10,20,30,40]})
sns.jointplot('x1','x2',df1)
pca = decomposition.PCA(n_components=1)
pca.fit(df1)
pca.components_[0]
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
df1_pca = pca.transform(df1)

#create dataframe with high correlation
df2 = pd.DataFrame({'x1':[10,20,30,40],'x2':[15,25,38,44]})
sns.jointplot('x1','x2',df2)
pca = decomposition.PCA(n_components=1)
pca.fit(df2)
pca.components_[0]
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
df2_pca = pca.transform(df2)