from flask import Flask, request, jsonify
import pandas as pd

from trie.keyboard import create_keyboard
from trie.trie import Node, insert_key
from trie.predict import predict

from clustering.TCluster import TCluster

app = Flask(__name__)


keyboard = create_keyboard('data/keyboard2.txt')

df_training = pd.read_excel('data/wordFrequency.xlsx', sheet_name='4 forms (219k)')

training_words = df_training['word'].tolist() #+ df_vocab['word'].tolist()

# Filter only the words that are alpha
training_words = [str(word).lower() for word in training_words if str(word).isalpha()]

# Create the trie
root = Node()

for word in training_words:
    insert_key(root, word)



@app.route('/predict', methods=['POST'])
def predict_word():
    data = request.json

    points = data['gaze_points']

    points = [(point['x'], point['y'], point['time']) for point in points if point['y'] > 0]

    result = predict(points, keyboard, root)

    print(result)

    return jsonify({'top_words': result})


@app.route('/cluster', methods=['POST'])
def predict_cluster():
    data = request.json

    points = data['gaze_points']

    points = [(point['x'], point['y'], point['time']) for point in points if point['y'] > 0]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster()
    tc.fit(df)

    keys = tc.predict(keyboard, root)

    return jsonify({'top_words': keys})




@app.route('/test', methods=['POST'])
def testing():

    df = pd.read_csv('data/collection_v2.csv')
    df = df.groupby('word_id')

    tc = TCluster()

    results = []

    for word_id, group in df:
        tc.fit(group[['x', 'y', 'time']])
        keys = tc.predict(keyboard, root)

        results.append({'word_id': word_id, 'keys': keys})
    
    return jsonify({'results': results})




if __name__ == '__main__':
    app.run(port=5000, debug=True)
