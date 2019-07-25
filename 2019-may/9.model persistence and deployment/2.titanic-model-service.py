import flask
import pandas as pd
from sklearn.externals import joblib
import common_utils as utils
import os


app = flask.Flask("test_service")

@app.route('/titanic/predict/', methods=['POST'])
def predict() :
    data = flask.request.json
    print("in service")
    print(data)
    titanic_test = pd.DataFrame(data)
    #print(titanic_test.info()) 
    
    model_objects = joblib.load(os.path.join(dir,'titanic_model_1.pkl') )
    
    titanic_test1 = utils.drop_features(titanic_test, ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    utils.cast_to_cat(titanic_test1, ['Sex', 'Pclass', 'Embarked'])

    cat_features = utils.get_categorical_features(titanic_test1)
    #print(cat_features)
    cont_features = utils.get_continuous_features(titanic_test1)
    #print(cont_features)

    titanic_test1[cat_features] = model_objects.get('cat_imputers').transform(titanic_test1[cat_features])
    titanic_test1[cont_features] = model_objects.get('cont_imputers').transform(titanic_test1[cont_features])

    utils.cast_to_cat(titanic_test1, ['Sex', 'Pclass', 'Embarked'])

    titanic_test1['Sex'] = titanic_test1['Sex'].cat.add_categories(['male', 'female'])
    titanic_test1['Pclass'] = titanic_test1['Pclass'].cat.add_categories([1,2,3])
    titanic_test1['Embarked'] = titanic_test1['Embarked'].cat.add_categories(['S','Q','C'])
    #print(titanic_test1.info())

    titanic_test2 = utils.ohe(titanic_test1, cat_features)
    print(titanic_test2.shape)
    X_test = model_objects.get('scaler').transform(titanic_test2)
    result = model_objects.get('estimator').predict(X_test)
    print(result)
    return flask.jsonify(prediction=str(1))  


if __name__ == '__main__':
       app.run(port=8080)