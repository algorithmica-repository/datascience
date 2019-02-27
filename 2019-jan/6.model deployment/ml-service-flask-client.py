import requests

url =  'http://localhost:8080/predict/'
data = [{ "Age":20, "PassengerId":100, "Sex":"male", "Fare":34.6, "Pclass":3, "Embarked":"S", "Parch":2, "SibSp":3, "Ticket":"XXX", "Cabin":"X"},
        { "Age":25, "PassengerId":200, "Sex":"female", "Fare":38.6, "Pclass":2, "Embarked":"S", "Parch":2, "SibSp":3, "Ticket":"XXX", "Cabin":"X"}
        ]
r = requests.post(url,json=data)
print(r.json())


#data = json.dumps([{}, {}])