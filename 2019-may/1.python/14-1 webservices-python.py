import flask

app = flask.Flask("test_service")

@app.route('/predict1/', methods=['GET'])
def test1() :
    return 'hello1'

@app.route('/predict2/', methods=['GET'])
def test2() :
    return 'hello2'

if __name__ == '__main__':
       app.run(port=8080)