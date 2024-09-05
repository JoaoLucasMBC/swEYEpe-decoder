import os
import json
from collections import defaultdict
import re
from tqdm import tqdm

def generate_bigrams(corpus):
    """
    Generates bigrams from the given corpus and calculates their probabilities.

    @params:
    corpus: list[str] - A list of sentences to create bigrams from.

    @returns:
    bigram_probs: dict - A dictionary where each key is a bigram and the value is a dict of next word probabilities.
    """
    
    # Initialize defaultdict for bigram counts
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    # Use tqdm for progress bar
    for sentence in tqdm(corpus, desc="Processing sentences"):
        words = ['<s>', "<s>"] + clean_sentence(sentence)

        for i in range(len(words) - 2):
            bigram = ' '.join(words[i:i+2])
            next_word = words[i+2]
            bigram_counts[bigram][next_word] += 1
    
    # Convert counts to probabilities
    bigram_probs = {}
    for bigram, next_words in bigram_counts.items():
        total_count = sum(next_words.values())
        bigram_probs[bigram] = {word: count / total_count for word, count in next_words.items()}
    
    return bigram_probs



def clean_sentence(sentence):
    """
    Cleans a sentence by lowercasing it, removing non-alphabetic characters,
    IDs, and tags like <h> and <p>.

    @params:
    sentence: str - The sentence to clean.

    @returns:
    list[str] - A list of cleaned words.
    """
    # Remove IDs starting with @@ and tags like <h> and <p>
    sentence = re.sub(r'@@\d+', '', sentence)  # Remove IDs like @@1234567
    sentence = re.sub(r'<[^>]+>', '', sentence)  # Remove tags like <h> or <p>

    # Lowercase the sentence and remove punctuation (except single quotes within words)
    sentence = sentence.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for p in punctuation:
        sentence = sentence.replace(p, '')
    
    if sentence and sentence.strip():
        # Return only alphabetic words
        return [word for word in sentence.split() if word.isalpha()]

    return []


def save_bigram_probs(bigram_probs, file_path):
    """
    Saves the bigram probability table to a JSON file.

    @params:
    bigram_probs: dict - The bigram probability table.
    file_path: str - The path where to save the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(bigram_probs, f, indent=4)

def read_files_from_folder(folder_path):
    """
    Reads all files from a specified folder and splits text into sentences.
    Sentences are split both by line breaks and periods ('.').

    @params:
    folder_path: str - The path to the folder containing the text files.

    @returns:
    corpus: list[str] - A list of sentences from all the files in the folder.
    """
    corpus = []
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Split the line by periods to handle sentence breaks
                    sentences = line.split(".")
                    
                    corpus.extend(sentences)
    
    return corpus

# Example usage
if __name__ == "__main__":
    import sys

    folder_path = sys.argv[1]

    if folder_path is None:
        print("Usage: python generate_bigrams.py <folder_path>")
        sys.exit(1)
    
    corpus = read_files_from_folder(folder_path)
    bigram_probs = generate_bigrams(corpus)
    save_bigram_probs(bigram_probs, 'bigram_v2.json')
    print("Bigram probability table saved to 'bigram.json'")
