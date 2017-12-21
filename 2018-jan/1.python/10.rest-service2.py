from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def greet():
    data = request.json
    print(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=8080)