from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     prediction = dt_estimator.predict(query_df)
     return jsonify({'prediction': list(prediction)})
 
MODEL_DIR = 'C:/Users/Algorithmica/Downloads'
MODEL_FILE = 'decision-tree-v1.pkl'
if __name__ == '__main__':
     os.chdir(MODEL_DIR)
     dt_estimator = joblib.load(MODEL_FILE)
     app.run(port=8080)