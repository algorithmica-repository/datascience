from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def greet():
   return 'hello flask' 

@app.route('/home', methods=['GET'])
def home():
   return 'hello home' 


if __name__ == '__main__':
    app.run(port=8080)