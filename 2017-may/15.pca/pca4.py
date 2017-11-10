from sklearn import decomposition, preprocessing
import pandas as pd

#PCA without standardization
sqft = [1000, 600, 1200, 1400]
n_rooms = [3, 2, 2, 4]
price = [1000000, 700000, 1200000, 1500000]
house_data = pd.DataFrame({'sqft':sqft, 'n_rooms':n_rooms, 'price':price})
pca = decomposition.PCA()
pca.fit(house_data)
pca.explained_variance_
pca.explained_variance_ratio_

#PCA with standardization
scaler = preprocessing.StandardScaler()
scaler.fit(house_data)
house_data1 = scaler.transform(house_data)
pca.fit(house_data1)
pca.explained_variance_
pca.explained_variance_ratio_

