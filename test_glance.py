import json
import numpy as np
from vis.glance import GlanceWriterDecoder

def parse_keyboard_layout(layout_str):
    """Parse the keyboard layout string into the required format."""
    # Parse the layout string
    layout = eval(layout_str)
    
    # Extract key positions
    key_positions = {}
    for key_pos in layout['keyboard'].split('\n'):
        key, pos = key_pos.split(' (')
        pos = pos.rstrip(')')
        x, y = map(float, pos.split(','))
        # Create key points (center and vertices for a circular key)
        center = (x, y)
        radius = 0.05  # Approximate key size
        key_positions[key] = [
            center,  # center
            (x - radius, y - radius),  # top_left
            (x + radius, y - radius),  # top_right
            (x + radius, y + radius),  # bottom_right
            (x - radius, y + radius)   # bottom_left
        ]
    
    return key_positions

def load_eye_data(file_path):
    """Load and parse eye tracking data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract gaze points
    gaze_points = []
    for entry in data:
        if 'input_data' in entry and 'gaze_points' in entry['input_data']:
            for point in entry['input_data']['gaze_points']:
                gaze_points.append((point['x'], point['y'], point['z']))
    
    return gaze_points

def main():
    # Load keyboard layout
    with open('vis/circleLayout.txt', 'r') as f:
        layout_str = f.read()
    key_positions = parse_keyboard_layout(layout_str)
    
    # Load eye tracking data
    eye_data_path = 'No-Delete-Study/No-Delete-Study/Participant 1 (Joseph)/Day 1 Circle/2025-04-29-17-39_TutorialSwiperProgression_eyeData.txt'
    gaze_points = load_eye_data(eye_data_path)
    
    # Create a simple dictionary for testing
    dictionary = ['hello', 'world', 'test', 'circle', 'keyboard', 'eye', 'tracking']
    
    # Initialize decoder
    decoder = GlanceWriterDecoder(
        key_positions=key_positions,
        dictionary=dictionary,
        sigma=0.4,
        stability_window=30,
        eps=0.07,
        min_samples=5
    )
    
    # Process gaze points in chunks
    chunk_size = 100
    for i in range(0, len(gaze_points), chunk_size):
        chunk = gaze_points[i:i + chunk_size]
        predictions = decoder.decode(chunk, top_k=3)
        print(f"\nChunk {i//chunk_size + 1} predictions:")
        for word, score in predictions:
            print(f"{word}: {score:.4f}")

if __name__ == "__main__":
    main() 