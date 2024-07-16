from flask import Flask, request, jsonify
from main import start

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    print(data)

    points = data['gaze_points']

    print(points)

    result = start(points)
    
    print(result)

    return jsonify({'top_words': result})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
