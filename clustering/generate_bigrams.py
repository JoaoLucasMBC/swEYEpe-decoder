import json
from collections import defaultdict

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
    
    for sentence in corpus:
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
    Cleans a sentence by lowercasing it and removing non-alphabetic characters.

    @params:
    sentence: str - The sentence to clean.

    @returns:
    list[str] - A list of cleaned words.
    """
    sentence = sentence.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for p in punctuation:
        sentence = sentence.replace(p, ' ')
    
    return [word for word in sentence.split() if word.isalpha()]


def save_bigram_probs(bigram_probs, file_path):
    """
    Saves the bigram probability table to a JSON file.

    @params:
    bigram_probs: dict - The bigram probability table.
    file_path: str - The path where to save the JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(bigram_probs, f, indent=4)

# Example usage
if __name__ == "__main__":
    import sys

    path = sys.argv[1]

    if path is None:
        print("Usage: python generate_bigrams.py <corpus_path>")
        sys.exit(1)
    

    if path == "example":
        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The dog barked at the fox",
            "Foxes are quick and clever animals",
            "The lazy dog did not care about the quick fox"
        ]
        
        bigram_probs = generate_bigrams(corpus)
        save_bigram_probs(bigram_probs, '../data/bigram.json')
        print("Bigram probability table saved to 'bigram.json'")

    else:
        with open(path, 'r') as f:
            corpus = f.readlines()
        
        bigram_probs = generate_bigrams(corpus)
        save_bigram_probs(bigram_probs, 'bigram.json')
        print("Bigram probability table saved to 'bigram.json'")