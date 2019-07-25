import requests

url =  'http://localhost:8080/predict1/'
r = requests.get(url)
print(type(r))
print(r.content)
