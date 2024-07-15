from flask import Flask, request, jsonify
from main import start

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    points = data['gaze_points']

    result = start(points)

    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
