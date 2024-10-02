import math
import json
import re
from collections import defaultdict

class Key:
    def __init__(self, letter, center, width, height):
        self.letter = letter
        self.center = center
        self.width = width
        self.height = height

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None
        self.key_score = 0
        self.sum_score = 0
        self.state = "RELEASE"

class GlanceWriterDecoder:
    def __init__(self, keyboard_layout):
        self.root = TrieNode()
        self.hold_nodes = []
        self.word_candidates = {}
        self.time_limit = 5  # Time limit to hold a word in candidates
        self.window_size = 30  # Size of the window for gaze stability
        self.sigma = 0.4  # Standard deviation for Gaussian distribution
        self.keyboard = keyboard_layout

    def insert_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word

    def calculate_distance_score(self, distance):
        return (1 / (self.sigma * math.sqrt(2 * math.pi))) * math.exp(-(distance ** 2) / (2 * self.sigma ** 2))

    def calculate_stability_score(self, gaze_points):
        if len(gaze_points) < 2:
            return 1
        speeds = [self.calculate_speed(gaze_points[i], gaze_points[i+1]) for i in range(len(gaze_points) - 1)]
        avg_speed = sum(speeds) / len(speeds)
        return 1 / (avg_speed + 1e-6)  # Adding small value to avoid division by zero

    def calculate_speed(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def calculate_key_score(self, gaze_points, key):
        max_score = 0
        for point in gaze_points:
            distance = self.calculate_speed(point, key.center)
            distance_score = self.calculate_distance_score(distance)
            stability_score = self.calculate_stability_score(gaze_points[-self.window_size:])
            score = distance_score * stability_score
            max_score = max(max_score, score)
        return max_score

    def update_hold_nodes(self, current_key, key_score):
        new_hold_nodes = []
        for node in self.hold_nodes:
            if current_key.letter in node.children:
                child = node.children[current_key.letter]
                if child.state == "RELEASE":
                    child.state = "HOLD"
                    child.key_score = key_score
                    child.sum_score = node.sum_score + key_score
                    new_hold_nodes.append(child)
                else:
                    if key_score > child.key_score:
                        child.key_score = key_score
                        child.sum_score = node.sum_score + key_score
                    new_hold_nodes.append(child)
        
        # Add nodes starting with current_key
        if current_key.letter in self.root.children:
            node = self.root.children[current_key.letter]
            node.state = "HOLD"
            node.key_score = key_score
            node.sum_score = key_score
            new_hold_nodes.append(node)
        
        self.hold_nodes = sorted(new_hold_nodes, key=lambda x: x.sum_score, reverse=True)[:50]

    def update_word_candidates(self):
        current_time = 0  # This should be replaced with actual timestamp
        for node in self.hold_nodes:
            if node.is_word:
                self.word_candidates[node.word] = (node.sum_score, current_time)
        
        # Remove old candidates
        self.word_candidates = {word: (score, time) for word, (score, time) in self.word_candidates.items() 
                                if current_time - time <= self.time_limit}

    def get_key_from_gaze(self, gaze_point):
        for key in self.keyboard:
            if (abs(gaze_point[0] - key.center[0]) <= key.width / 2 and
                abs(gaze_point[1] - key.center[1]) <= key.height / 2):
                return key
        return None  # Return None if no key is found

    def decode_gaze_path(self, gaze_path):
        for gaze_point in gaze_path:
            current_key = self.get_key_from_gaze(gaze_point)
            if current_key:
                key_score = self.calculate_key_score([gaze_point], current_key)
                self.update_hold_nodes(current_key, key_score)
                self.update_word_candidates()
        
        # Sort candidates by score and return top 5
        sorted_candidates = sorted(self.word_candidates.items(), key=lambda x: x[1][0], reverse=True)
        return [word for word, _ in sorted_candidates[:5]]

# Usage example
def create_keyboard_layout():
    # This is a simplified QWERTY layout. Adjust key positions and sizes as needed.
    keys = [
        Key('Q', (10, 10), 20, 20), Key('W', (30, 10), 20, 20), Key('E', (50, 10), 20, 20),
        Key('R', (70, 10), 20, 20), Key('T', (90, 10), 20, 20), Key('Y', (110, 10), 20, 20),
        Key('U', (130, 10), 20, 20), Key('I', (150, 10), 20, 20), Key('O', (170, 10), 20, 20),
        Key('P', (190, 10), 20, 20),
        Key('A', (20, 30), 20, 20), Key('S', (40, 30), 20, 20), Key('D', (60, 30), 20, 20),
        Key('F', (80, 30), 20, 20), Key('G', (100, 30), 20, 20), Key('H', (120, 30), 20, 20),
        Key('J', (140, 30), 20, 20), Key('K', (160, 30), 20, 20), Key('L', (180, 30), 20, 20),
        Key('Z', (30, 50), 20, 20), Key('X', (50, 50), 20, 20), Key('C', (70, 50), 20, 20),
        Key('V', (90, 50), 20, 20), Key('B', (110, 50), 20, 20), Key('N', (130, 50), 20, 20),
        Key('M', (150, 50), 20, 20)
    ]
    return keys

def correct_json(content):
    # Replace single quotes with double quotes
    content = content.replace("'", '"')
    
    # Correct boolean values
    content = content.replace('True', 'true').replace('False', 'false')
    
    # Remove trailing commas in lists and objects
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    
    # Wrap the entire content in curly braces if it's not already
    if not content.strip().startswith('{'):
        content = '{' + content + '}'
    
    return content

def read_keyboard_layout(layout_file):
    with open(layout_file, 'r') as f:
        data = f.read()
        corrected_data = correct_json(data)
        layout_data = json.loads(corrected_data)
    
    keyboard_str = layout_data['keyboard']
    center = layout_data['center']
    left_bound = layout_data['left_bound']
    right_bound = layout_data['right_bound']
    top_bound = layout_data['top_bound']
    bottom_bound = layout_data['bottom_bound']
    
    # Calculate the width and height of the entire keyboard
    keyboard_width = right_bound['x'] - left_bound['x']
    keyboard_height = top_bound['y'] - bottom_bound['y']
    
    # Calculate the width and height of each key (assuming uniform key size)
    key_width = keyboard_width / 10  # Assuming 10 keys in the longest row (QWERTYUIOP)
    key_height = keyboard_height / 3  # Assuming 3 rows of keys
    
    keys = []
    for line in keyboard_str.split('\n'):
        letter, pos = line.split(' ')
        x, y = map(float, pos.strip('()').split(','))
        
        # Convert the normalized coordinates to pixel coordinates
        pixel_x = (x - left_bound['x']) * (keyboard_width / 2)
        pixel_y = (y - bottom_bound['y']) * (keyboard_height / 2)
        
        keys.append(Key(letter, (pixel_x, pixel_y), key_width, key_height))
    
    return keys