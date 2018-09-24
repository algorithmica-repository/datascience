from sklearn import decomposition, manifold
import pandas as pd

df1= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[20,5,17,20,22],
        'F3':[10,2,7,10,11]})
print(df1.corr())

pca1 = decomposition.PCA(n_components=1)
pca1.fit(df1)
df2 = pca1.transform(df1)
print(pca1.explained_variance_)
print(pca1.explained_variance_ratio_)
print(pca1.components_)

tsne = manifold.TSNE()
df3 = tsne.fit_transform(df1)
print(tsne.embedding_)
print(tsne.kl_divergence_)
