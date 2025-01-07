# Standard library imports
import json
import os
from datetime import datetime

# Third-party imports
from flask import Flask, request, jsonify
import pandas as pd

# Local imports
from trie.keyboard import create_keyboard
from trie.trie import Node, insert_key
from trie.predict import predict
from clustering.TCluster import TCluster
from languageContext.LanguageContext import LanguageContext

app = Flask(__name__)

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

LC = LanguageContext()

bigram_path = os.path.join('data', 'bigram_v2.json')
# Context parameters for sentences bigrams
with open(bigram_path, 'r') as f:
    bigram_probs: dict[dict] = json.load(f)

vocab_path = os.path.join('data', 'vocab_final.csv')
vocab = pd.read_csv(vocab_path)
t = datetime.today().strftime('%Y-%m-%d %H-%M-%S')


@app.route('/setup', methods=['POST'])
def setup_keyboard():
    data = request.json
    global t
    t = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    # t = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    with open("layout.txt", 'w') as file:
        file.write(str(data))
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
    global LC
    tc = TCluster(K=number_of_letters_to_get, vocab=vocab, context_probs=None, eps=0.07)
    # tc = TCluster(K=number_of_letters_to_get, vocab=vocab)
    tc.fit(df)
    global custom_keyboard
    gaze_scores = tc.predict(custom_keyboard, root, allProbs = True)
    probs = [(key[0], float(key[1][0])) for key in gaze_scores]
    just_p = [key[1] for key in probs]
    tot = sum(just_p)
    gaze_probs = [(key[0], key[1]/tot) for key in probs]

    con = ""
    for word in context:
        con += word + " "
    print(con.strip())
    if (con.strip() != ""):
        language_scores = LC.words_and_probs(con.strip().lower())
        keys = LC.combine_probs(gaze_probs = gaze_probs, language_probs = language_scores, language_weight = 0.3)
    else:
        keys = gaze_scores[:3]
    if (keys == None):
        return jsonify({'top_words': ["i", "a", "is"]})
    try:
        
        global t
        with open("eyeData/eyeTracking" + t + ".txt", 'a') as file:
            file.write(str(data) + '\n')
            file.write(str({'top_words': [key[0] for key in keys]}) + "\n")
            # file.write(str(contextReal) + "\n")

        return jsonify({'top_words': [key[0] for key in keys]})
    except Exception as e:
        print(e)
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
