from flask import Flask, request, jsonify
import pandas as pd

from keyboard import create_keyboard
from trie import Node, insert_key


app = Flask(__name__)


keyboard = create_keyboard('../data/keyboard2.txt')

df_training = pd.read_excel('../data/wordFrequency.xlsx', sheet_name='4 forms (219k)')

training_words = df_training['word'].tolist() #+ df_vocab['word'].tolist()

# Filter only the words that are alpha
training_words = [str(word).lower() for word in training_words if str(word).isalpha()]

# Create the trie
root = Node()

for word in training_words:
    insert_key(root, word)



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    print(data)

    points = data['gaze_points']

    print(points)

    dt = 0.0166666667 # 60 fps
    time = 0

    for i in range(len(points)):
        x, y = points[i].values()
        points[i] = (x, y, time)
        time += dt


    result = predict(points, keyboard, root)

    print(result)

    return jsonify({'top_words': result})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
