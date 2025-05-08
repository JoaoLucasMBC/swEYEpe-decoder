import pandas as pd
import numpy as np
import sys
import glancewriterDecoder as gd
import re
import matplotlib.pyplot as plt
from importlib import reload
sys.path.append("..") #voodoo shit
from trie.keyboard import create_keyboard
from trie.trie import Node, insert_key
from trie.predict import predict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from clustering.TCluster import TCluster
import os
import json

def parse_no_delete_study(base_path="../No-Delete-Study"):
    """
    Parse the No-Delete-Study folder structure and organize data by participant, day, and keyboard type.
    Skips pilot folders and separates QWERTY and Circle keyboard data.
    
    Args:
        base_path (str): Path to the No-Delete-Study folder
        
    Returns:
        dict: Organized data structure with the following hierarchy:
            {
                'participant_id': {
                    'qwerty': {
                        'day_1': [file_paths],
                        'day_2': [file_paths],
                        ...
                    },
                    'circle': {
                        'day_1': [file_paths],
                        'day_2': [file_paths],
                        ...
                    }
                }
            }
    """
    study_data = {}
    
    # Navigate to the correct folder containing participant data
    study_path = os.path.join(base_path, "No-Delete-Study")
    
    # Iterate through participant folders
    for participant_folder in os.listdir(study_path):
        if not os.path.isdir(os.path.join(study_path, participant_folder)):
            continue
            
        participant_id = participant_folder
        study_data[participant_id] = {
            'qwerty': {},
            'circle': {}
        }
        
        participant_path = os.path.join(study_path, participant_folder)
        
        # Iterate through day folders
        for day_folder in os.listdir(participant_path):
            if 'Pilot' in day_folder:  # Skip pilot folders
                continue
                
            if not os.path.isdir(os.path.join(participant_path, day_folder)):
                continue
                
            # Determine keyboard type and day number
            if 'QWERTY' in day_folder:
                keyboard_type = 'qwerty'
            elif 'Circle' in day_folder:
                keyboard_type = 'circle'
            else:
                continue
                
            day_num = int(day_folder.split()[1])  # Extract day number
            day_key = f'day_{day_num}'
            
            # Initialize day list if it doesn't exist
            if day_key not in study_data[participant_id][keyboard_type]:
                study_data[participant_id][keyboard_type][day_key] = []
            
            # Get all files in the day folder
            day_path = os.path.join(participant_path, day_folder)
            for file_name in os.listdir(day_path):
                file_path = os.path.join(day_path, file_name)
                if os.path.isfile(file_path):
                    study_data[participant_id][keyboard_type][day_key].append(file_path)
    
    return study_data

# Example usage:
study_data = parse_no_delete_study()
print(json.dumps(study_data, indent=2))





