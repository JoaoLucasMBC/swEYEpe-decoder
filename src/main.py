from trie import Node, insert_key, print_trie
from keyboard import create_keyboard
from predict import predict

import pandas as pd

def main():
    with open('../data/eye-test.txt', 'r') as f:
        lines: list[str] = f.readlines()

    df_training = pd.read_excel('../data/wordFrequency.xlsx', sheet_name='4 forms (219k)')
    training_words = df_training['word'].tolist()

    # Filter only the words that are alpha
    training_words = [str(word).lower() for word in training_words if str(word).isalpha()]
    
    keyboard = create_keyboard('../data/keyboard.txt')

    # for each word, you have a list of tuples with all the coordinates of the eye while typing that word
    words = {}
    curr = ''

    for line in lines:
        line = line.replace('\n', '')
        if line:
            if line.isalpha():
                word = line.strip().lower()
                curr = word
                words[word] = []
            else:
                x, y = map(float, line.split(','))
                words[curr].append((x, y))
    
    # Create the trie
    root = Node()

    for word in training_words:
        insert_key(root, word)

    for word in words:
        predict(word, words[word], keyboard, root)


if __name__ == "__main__":
    main()