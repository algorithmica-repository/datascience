from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.json
    print(data)
    titanic_test = pd.DataFrame(data)
    print(titanic_test.info())

    imputable_cont_features = objects_map.get('imputable-features')
    cont_imputer = objects_map.get('cont-imputer')
    titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])

    cat_imputer = objects_map.get('cat-imputer')
    titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])

    titanic_test['FamilySize'] = titanic_test['SibSp'] +  titanic_test['Parch'] + 1
    titanic_test['FamilyGroup'] = titanic_test['FamilySize'].map(objects_map.get('family-size-func'))

    titanic_test['Sex'] = objects_map.get('le-sex').transform(titanic_test['Sex'])
    titanic_test['Embarked'] = objects_map.get('le-emb').transform(titanic_test['Embarked'])
    titanic_test['Pclass'] = objects_map.get('le-pclass').transform(titanic_test['Pclass'])
    titanic_test['FamilyGroup'] = objects_map.get('le-fgroup').transform(titanic_test['FamilyGroup'])

    cat_features = objects_map.get('cat-features')
    tmp1 = objects_map.get('ohe').transform(titanic_test[cat_features]).toarray()

    cont_features = objects_map.get('cont-features')
    tmp2 = titanic_test[cont_features].values
    X_test = np.concatenate((tmp1,tmp2), axis=1)
    print(X_test)
    estimator = objects_map.get('estimator')
    predictions = estimator.predict(X_test)
    print(predictions)
    return jsonify(prediction=str(predictions))

if __name__ == '__main__':
    path = 'C:\\Users\\Algorithmica\\Downloads'
    objects_map = joblib.load( os.path.join(path, 'deployment.pkl') )
    app.run(port=8080)