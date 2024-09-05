from flask import Flask, request, jsonify
import pandas as pd

from trie.keyboard import create_keyboard
from trie.trie import Node, insert_key
from trie.predict import predict

from clustering.TCluster import TCluster

import os
import json

app = Flask(__name__)


keyboard = create_keyboard('data/keyboard/keyboard2.txt')
keyboard_circle = create_keyboard('data/keyboard/keyboard_circle.txt')
keyboard_circle_alpha = create_keyboard('data/keyboard/keyboard_circle_alphabetical.txt')

different_letters_keyboard = create_keyboard('data/keyboard/test_letter_differentiation copy.txt')
keyboard_26_circle = create_keyboard('data/keyboard/keyboard_26_sections.txt')

#df_training = pd.read_excel('data/wordFrequency.xlsx', sheet_name='4 forms (219k)')
df_training = pd.read_csv('data/vocab_final.csv')

training_words = df_training['word'].tolist()

# Filter only the words that are alpha
training_words = [str(word).lower() for word in training_words if str(word).isalpha()]

# Create the trie
root = Node()

for word in training_words:
    insert_key(root, word)

custom_keyboard = create_keyboard('data/keyboard/keyboard2.txt')
custom_inner_radius = 0
custom_outer_radius = 0
custom_center = (0, 0)
number_of_letters_to_get = 1
keyboard_shape = ""

bigram_path = os.path.join('data', 'bigram_v2.json')
# Context parameters for sentences bigrams
with open(bigram_path, 'r') as f:
    bigram_probs: dict[dict] = json.load(f)

vocab_path = os.path.join('data', 'vocab_final.csv')
vocab = pd.read_csv(vocab_path)


@app.route('/setup', methods=['POST'])
def setup_keyboard():
    data = request.json
    # print(data)
    global custom_keyboard 
    global custom_center
    global custom_inner_radius
    global custom_outer_radius
    global number_of_letters_to_get
    global keyboard_shape
    global top_bound, bottom_bound, left_bound, right_bound
    right_bound = (data['right_bound']['x'], data['right_bound']['y'])
    left_bound = (data['left_bound']['x'], data['left_bound']['y'])
    top_bound = (data['top_bound']['x'], data['top_bound']['y'])
    bottom_bound = (data['bottom_bound']['x'], data['bottom_bound']['y'])
    print("Recieved new keyboard!")
    number_of_letters_to_get = data["k"]
    keyboard_shape = data["shape"]
    custom_keyboard = create_keyboard(data["keyboard"], useString=True)
    custom_center = (data['center']['x'], data['center']['y'])
    custom_inner_radius = data["inner_radius"]
    custom_outer_radius = data["outer_radius"]
    return jsonify({"message": "setup done!"})

@app.route('/circle', methods=['POST'])
def predict_circle():
    data = request.json

    points = data['gaze_points']
    radius = data['radius']
    center = (data['center']['x'], data['center']['y'])

    # Filter OUT the points that are in the circle
    points = [(point['x'], point['y'], point['z']) for point in points if (point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster(K=1)
    tc.fit(df)

    keys = tc.predict(keyboard_circle, root)

    return jsonify({'top_words': [key[0] for key in keys]})

@app.route('/circleOuter', methods=['POST'])
def predict_circle_outer():
    data = request.json
    print(jsonify(data))
    points = data['gaze_points']
    radius = data['radius'] #0.75
    outerRadius = data['outer_radius']
    center = (data['center']['x'], data['center']['y']) #0.525

    # Filter OUT the points that are in the circle
    points = [(point['x'], point['y'], point['z']) for point in points if ((point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2 and 
                                                                           (point['x'] - center[0])**2 + (point['y'] - center[1])**2 < outerRadius**2)]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster(K=1)
    tc.fit(df, verbose=True)

    keys = tc.predict(keyboard_circle_alpha, root, verbose=True)

    return jsonify({'top_words': [key[0] for key in keys]})

@app.route('/general', methods=['POST'])
def predict_general():
    data = request.json
    #print(data)
    # custom_keyboard = create_keyboard(data["keyboard"], useString=True)
    
    points = data['gaze_points']
    global custom_outer_radius
    global custom_inner_radius
    radius = custom_inner_radius
    outerRadius = custom_outer_radius
    global custom_center
    center = custom_center
    global number_of_letters_to_get
    global top_bound, bottom_bound, left_bound, right_bound
    global keyboard_shape
    if (keyboard_shape == "circle"):
        # Filter OUT the points that are in the inner circle and outside the outer circle
        # print("circle")
        points = [(point['x'], point['y'], point['z']) for point in points if ((point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2 and 
                                                                           (point['x'] - center[0])**2 + (point['y'] - center[1])**2 < outerRadius**2)]
    if (keyboard_shape == "rectangle"):
        # Filter out points that are not in the rectangle
        # print("rectangle")
        points = [(point['x'], point['y'], point['z']) for point in points if ((point['x'] > left_bound[0]) and (point['x'] < right_bound[0]) and (point['y'] > bottom_bound[1]) and (point['y'] < top_bound[1]))]
    #print("post-filter data")
    #print(points)

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])
    context = data.get('context', [])
    context = data.get('asdfasdfasdf', []) # here to not get the context because its breaking the decoder
    if len(context) < 2:
        last_two: list[str] = ['<s>', '<s>']
    else:
        last_two: list[str] = context[-2:]

    context_probs: dict[dict] = bigram_probs.get(' '.join(last_two), {})

    try:
        tc = TCluster(K=number_of_letters_to_get, vocab=vocab, context_probs=context_probs)
        tc.fit(df)
        global custom_keyboard
        keys = tc.predict(custom_keyboard, root)
        if (keys == None):
            return jsonify({'top_words': ["i", "a", "is"]})

        return jsonify({'top_words': [key[0] for key in keys]})
    except:
        return jsonify({'top_words': ["i", "a", "is"]})





@app.route('/test', methods=['POST'])
def testing():

    df = pd.read_csv('data/user/collection_v2.csv')
    df = df.groupby('word_id')

    tc = TCluster()

    results = []

    for word_id, group in df:
        group = group[group['y'] > 0]

        if len(group) == 0:
            continue
        
        tc.fit(group[['x', 'y', 'time']])
        keys = tc.predict(keyboard, root)

        results.append({'word_id': word_id, 'keys': keys})
    
    return jsonify({'results': results})




if __name__ == '__main__':
    app.run(port=5000, debug=True)
