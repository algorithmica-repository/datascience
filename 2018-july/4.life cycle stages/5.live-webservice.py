from sklearn.externals import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/', methods=['GET'])
def default():
    return "hello"
    
@app.route('/titanic', methods=['POST'])
def get_prediction():
     data = request.json
     titanic_test = pd.DataFrame(data)
     print(titanic_test.shape)
     print(titanic_test.info())
     titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

     titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
     titanic_test1.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin'], axis=1, inplace=True)
     predictions = classifier.predict(titanic_test1)
     print(predictions)
     return jsonify(prediction=predictions)

@app.route('/titanic', methods=['GET'])
def get_all_predictions():
     titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
     print(titanic_test.shape)
     print(titanic_test.info())
     titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

     titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
     titanic_test1.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin'], axis=1, inplace=True)
     predictions = classifier.predict(titanic_test1)
     print(predictions)
     return predictions


if __name__ == "__main__":
    cat_columns = joblib.load("C:/Users/Algorithmica/Downloads/all/features_v1.pkl");
    classifier = joblib.load("C:/Users/Algorithmica/Downloads/all/titanic_dt_v1.pkl");
    app.run(port=80)
