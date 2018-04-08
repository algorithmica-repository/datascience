from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     data = request.json
     titanic_test = pd.DataFrame(data)
     features = ['Fare','SibSp', 'Parch']
     prediction = dt_estimator.predict(titanic_test1)
     print(prediction)
     return jsonify(prediction=prediction)
 
MODEL_DIR = 'C:/Users/Algorithmica/Downloads'
MODEL_FILE = 'decision-tree-v1.pkl'
if __name__ == '__main__':
     os.chdir(MODEL_DIR)
     dt_estimator = joblib.load(MODEL_FILE)
     app.run(port=8080)