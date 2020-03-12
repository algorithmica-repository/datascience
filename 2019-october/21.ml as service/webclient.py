import requests
import pandas as pd
import os

path = 'F://house-prices'
house_test = pd.read_csv(os.path.join(path,"test.csv"))
house_test.shape
house_test.info()
house_test['SalePrice'] = None

house_test1 = house_test.iloc[0:1,]
data = house_test1.to_json(orient='records')

url =  'http://localhost:8080/price/predict/'
r = requests.post(url, json=data)
print(r.json())