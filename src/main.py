from trie import Node, insert_key, print_trie, search_trie
from keyboard import create_keyboard
from predict import predict

import pandas as pd

def main():
    import sys

    if len(sys.argv) < 2:
        print('Usage: python main.py <path_to_file>')
        sys.exit(1)

    path = sys.argv[1]

    with open(path, 'r') as f:
        lines: list[str] = f.readlines()

    df_training = pd.read_excel('../data/wordFrequency.xlsx', sheet_name='4 forms (219k)')

    training_words = df_training['word'].tolist() #+ df_vocab['word'].tolist()

    # Filter only the words that are alpha
    training_words = [str(word).lower() for word in training_words if str(word).isalpha()]
    
    keyboard = create_keyboard('../data/keyboard2.txt')

    # for each word, you have a list of tuples with all the coordinates of the eye while typing that word
    words = {}
    curr = ''

    dt = 0.0166666667 # 60 fps
    time = 0
    i = 0

    # Parse the file and create the dictionary of words with the list of coordinates and the time for each point
    for line in lines:
        line = line.replace('\n', '')
        if line:
            if line.isalpha():
                word = line.strip().lower()
                curr = word + str(i)
                i += 1
                words[curr] = []
                time = 0
            else:
                x, y = map(float, line.split(','))
                words[curr].append((x, y, time))
                time += dt
    
    # Create the trie
    root = Node()

    for word in training_words:
        insert_key(root, word)

    for word in words:
        predict(words[word], keyboard, root, verbose=True)

    
if __name__ == "__main__":
    main()
