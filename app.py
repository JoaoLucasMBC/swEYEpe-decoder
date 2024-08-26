from flask import Flask, request, jsonify
import pandas as pd

from trie.keyboard import create_keyboard
from trie.trie import Node, insert_key
from trie.predict import predict

from clustering.TCluster import TCluster

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



@app.route('/predict', methods=['POST'])
def predict_word():
    data = request.json

    points = data['gaze_points']

    points = [(point['x'], point['y'], point['z']) for point in points if point['y'] > 0]

    result = predict(points, keyboard, root)

    print(result)

    return jsonify({'top_words': result})


@app.route('/cluster', methods=['POST'])
def predict_cluster():
    data = request.json

    points = data['gaze_points']

    points = [(point['x'], point['y'], point['z']) for point in points if point['y'] > 0]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster()
    tc.fit(df)

    keys = tc.predict(keyboard, root)

    return jsonify({'top_words': [key[0] for key in keys]})


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

@app.route('/circleAlpha', methods=['POST'])
def predict_circle_alpha():
    data = request.json

    points = data['gaze_points']
    radius = data['radius'] #0.75
    center = (data['center']['x'], data['center']['y']) #0.525

    # Filter OUT the points that are in the circle
    points = [(point['x'], point['y'], point['z']) for point in points if (point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster(K=1)
    tc.fit(df)

    keys = tc.predict(keyboard_circle_alpha, root)

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

@app.route('/circleSmaller', methods=['POST'])
def predict_circle_closer_smaller():
    data = request.json
    print(data)
    points = data['gaze_points']
    radius = data['radius'] #0.75
    outerRadius = data['outer_radius']
    center = (data['center']['x'], data['center']['y']) #0.525

    # Filter OUT the points that are in the circle
    points = [(point['x'], point['y'], point['z']) for point in points if ((point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2 and 
                                                                           (point['x'] - center[0])**2 + (point['y'] - center[1])**2 < outerRadius**2)]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster(K=1)
    tc.fit(df)

    keys = tc.predict(different_letters_keyboard, root)

    return jsonify({'top_words': [key[0] for key in keys]})

@app.route('/circle26', methods=['POST'])
def predict_circle26():
    data = request.json
    print(data)
    points = data['gaze_points']
    radius = data['radius'] #0.75
    outerRadius = data['outer_radius']
    center = (data['center']['x'], data['center']['y']) #0.525

    # Filter OUT the points that are in the circle
    points = [(point['x'], point['y'], point['z']) for point in points if ((point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2 and 
                                                                           (point['x'] - center[0])**2 + (point['y'] - center[1])**2 < outerRadius**2)]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster(K=1)
    tc.fit(df)

    keys = tc.predict(keyboard_26_circle, root)

    return jsonify({'top_words': [key[0] for key in keys]})

@app.route('/general', methods=['POST'])
def predict_general():
    data = request.json
    print(data)
    custom_keyboard = create_keyboard(data["keyboard"], useString=True)
    points = data['gaze_points']
    radius = data['radius'] #0.75
    outerRadius = data['outer_radius']
    center = (data['center']['x'], data['center']['y']) #0.525

    # Filter OUT the points that are in the circle
    points = [(point['x'], point['y'], point['z']) for point in points if ((point['x'] - center[0])**2 + (point['y'] - center[1])**2 > radius**2 and 
                                                                           (point['x'] - center[0])**2 + (point['y'] - center[1])**2 < outerRadius**2)]

    df = pd.DataFrame(points, columns=['x', 'y', 'time'])

    tc = TCluster(K=1)
    tc.fit(df)

    keys = tc.predict(custom_keyboard, root)

    return jsonify({'top_words': [key[0] for key in keys]})

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
