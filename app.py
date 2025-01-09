"""
VR Eye-Typing Flask Application
Handles eye tracking data processing and word prediction for VR typing interface.
"""

import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from flask import Flask, request, jsonify
import pandas as pd

from trie.keyboard import create_keyboard
from trie.trie import Node, insert_key
from clustering.TCluster import TCluster
from languageContext.LanguageContext import LanguageContext

# Type aliases
Point = Tuple[float, float, float]
GazePoint = Dict[str, float]

@dataclass
class KeyboardConfig:
    """Configuration for keyboard layout and parameters."""
    shape: str
    center: Tuple[float, float]
    inner_radius: float
    outer_radius: float
    k_letters: int  # number of letters to get
    bounds: Dict[str, Tuple[float, float]]

class EyeTypingApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.initialize_models()
        self.keyboard_config = None
        self.session_timestamp = None

    def setup_routes(self):
        """Configure Flask routes."""
        self.app.route('/setup', methods=['POST'])(self.setup_keyboard)
        self.app.route('/general', methods=['POST'])(self.predict_general)
        self.app.route('/test', methods=['POST'])(self.testing)

    def initialize_models(self):
        """Initialize ML models and data structures."""
        # Load vocabulary once
        vocab_path = os.path.join('data', 'vocab_final.csv')
        self.vocab_df = pd.read_csv(vocab_path)
        self.vocab = self.vocab_df  # Keep reference for TCluster, not really necessary right now because TC uses it for frequency, which is currently not used.
        self.training_words = self._load_training_words()
        
        # Initialize models
        self.root = self._build_trie()
        self.language_context = LanguageContext()
        self.custom_keyboard = None

    def _load_training_words(self) -> List[str]:
        """Process vocabulary into training words."""
        words = self.vocab_df['word'].tolist()
        return [str(word).lower() for word in words if str(word).isalpha()]

    def _build_trie(self) -> Node:
        """Build trie data structure from training words."""
        root = Node()
        for word in self.training_words:
            insert_key(root, word)
        return root

    def setup_keyboard(self):
        """Handle keyboard setup request."""
        data = request.json
        self.session_timestamp = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        
        # Save layout configuration
        self._save_layout_config(data)
        
        # Update keyboard configuration
        self.keyboard_config = KeyboardConfig(
            shape=data["shape"],
            center=(data['center']['x'], data['center']['y']),
            inner_radius=data["inner_radius"],
            outer_radius=data["outer_radius"],
            k_letters=data["k"],
            bounds={
                'top': (data['top_bound']['x'], data['top_bound']['y']),
                'bottom': (data['bottom_bound']['x'], data['bottom_bound']['y']),
                'left': (data['left_bound']['x'], data['left_bound']['y']),
                'right': (data['right_bound']['x'], data['right_bound']['y'])
            }
        )
        
        # Create keyboard layout
        self.custom_keyboard = create_keyboard(data["keyboard"], useString=True)
        
        return jsonify({"message": "Keyboard setup completed successfully"})

    def _save_layout_config(self, data: dict):
        """Save keyboard layout configuration to file."""
        try:
            with open("layout.txt", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving layout configuration: {str(e)}")

    def _filter_points(self, points: List[GazePoint]) -> List[Point]:
        """Filter gaze points based on keyboard shape and boundaries."""
        if self.keyboard_config.shape == "circle":
            return self._filter_circle_points(points)
        elif self.keyboard_config.shape == "rectangle":
            return self._filter_rectangle_points(points)
        return []

    def _filter_circle_points(self, points: List[GazePoint]) -> List[Point]:
        """Filter points for circular keyboard layout."""
        center = self.keyboard_config.center
        inner_r = self.keyboard_config.inner_radius
        outer_r = self.keyboard_config.outer_radius
        
        return [(p['x'], p['y'], p['z']) for p in points 
                if (inner_r**2 < (p['x'] - center[0])**2 + (p['y'] - center[1])**2 < outer_r**2)]

    def _filter_rectangle_points(self, points: List[GazePoint]) -> List[Point]:
        """Filter points for rectangular keyboard layout."""
        bounds = self.keyboard_config.bounds
        return [(p['x'], p['y'], p['z']) for p in points 
                if (bounds['left'][0] < p['x'] < bounds['right'][0] and 
                    bounds['bottom'][1] < p['y'] < bounds['top'][1])]

    def predict_general(self):
        """Handle prediction request for eye tracking data."""
        data = request.json
        self._save_incoming_data(data)
        
        # Process gaze points
        filtered_points = self._filter_points(data['gaze_points'])
        if not filtered_points:
            return jsonify({'top_words': ["i", "a", "is"]})
            
        df = pd.DataFrame(filtered_points, columns=['x', 'y', 'time'])
        
        # Initialize clustering
        tc = TCluster(
            K=self.keyboard_config.k_letters,
            vocab=self.vocab,
            context_probs=None,
            eps=0.07
        )
        
        # Get predictions
        tc.fit(df)
        gaze_scores = tc.predict(self.custom_keyboard, self.root, allProbs=True)
        
        # Process predictions with language model
        predictions = self._process_predictions(gaze_scores, data.get('context', []))
        
        # Save results
        self._save_prediction_results(data, predictions)
        
        return jsonify({'top_words': [key[0] for key in predictions]})

    def _process_predictions(self, gaze_scores, context):
        """Process gaze scores with language model context."""
        # Calculate gaze probabilities
        probs = [(key[0], float(key[1][0])) for key in gaze_scores]
        just_p = [key[1] for key in probs]
        tot = sum(just_p)
        gaze_probs = [(key[0], key[1]/tot) for key in probs]

        # If there's context, combine with language model
        context_str = " ".join(context).strip()
        if context_str:
            language_scores = self.language_context.words_and_probs(context_str.lower())
            return self.language_context.combine_probs(
                gaze_probs=gaze_probs,
                language_probs=language_scores,
                language_weight=0.5
            )
        return gaze_scores[:3]

    def _save_incoming_data(self, data: dict):
        """Save incoming request data."""
        try:
            with open("incoming.txt", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving incoming data: {str(e)}")

    def _save_prediction_results(self, data: dict, predictions: List):
        """Save prediction results and eye tracking data."""
        try:
            # Create directory if it doesn't exist
            os.makedirs('eyeData', exist_ok=True)
            
            # Construct the full entry with timestamp
            entry = {
                'timestamp': datetime.now().isoformat(),
                'input_data': data,
                'predictions': {
                    'top_words': [key[0] for key in predictions],
                    'scores': [float(key[1][0]) if isinstance(key[1], tuple) else float(key[1]) for key in predictions]
                }
            }

            filename = f"eyeData/eyeTracking{self.session_timestamp}.json"
            
            # Load existing data if file exists
            existing_data = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse existing file {filename}, starting fresh")
            
            # Append new entry
            if not isinstance(existing_data, list):
                existing_data = []
            existing_data.append(entry)
            
            # Write back to file
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving prediction results: {str(e)}")

    def testing(self):
        """Handle test endpoint for model evaluation."""
        df = pd.read_csv('data/user/collection_v2.csv')
        df = df.groupby('word_id')
        tc = TCluster()
        
        results = []
        for word_id, group in df:
            group = group[group['y'] > 0]
            if len(group) == 0:
                continue
            
            tc.fit(group[['x', 'y', 'time']])
            keys = tc.predict(self.custom_keyboard, self.root)
            results.append({'word_id': word_id, 'keys': keys})
        
        return jsonify({'results': results})

    def run(self, **kwargs):
        """Run the Flask application."""
        self.app.run(**kwargs)

if __name__ == '__main__':
    eye_typing_app = EyeTypingApp()
    eye_typing_app.run(port=5000, debug=True)