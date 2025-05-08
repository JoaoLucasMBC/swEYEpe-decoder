import re
import csv

def extract_unique_words(file_path):
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Convert to lowercase and split into words
                line_words = re.findall(r'\b\w+\b', line.lower())
                words.update(line_words)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()
    return sorted(words)

def get_vocab_words(file_path):
    vocab_words = set()
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:  # Ensure row has at least 2 columns
                    vocab_words.add(row[1].lower())  # word is in second column
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()
    return vocab_words

def print_sentences(file_path, num_sentences=100):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = []
            for i, line in enumerate(file):
                if i >= num_sentences:
                    break
                # Remove any trailing whitespace and add quotes
                sentence = line.strip()
                sentences.append(f'"{sentence}"')
            
            # Join sentences with commas
            print(', '.join(sentences))
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Get words from phrases
    phrase_words = set(extract_unique_words('data/phrases2.txt'))
    print(f"Found {len(phrase_words)} unique words in phrases")
    
    # Get words from vocab
    vocab_words = get_vocab_words('data/vocab_final.csv')
    print(f"Found {len(vocab_words)} words in vocab")
    
    # Find words in phrases but not in vocab
    missing_words = sorted(phrase_words - vocab_words)
    
    # Write to output file
    try:
        with open('missing_words.txt', 'w', encoding='utf-8') as output:
            for word in missing_words:
                output.write(f"{word}\n")
    except Exception as e:
        print(f"Error writing to missing_words.txt: {e}")
    
    print(f"Found {len(missing_words)} words in phrases that are not in vocab_final.csv")
    print("Words have been written to missing_words.txt")

    print_sentences('data/phrases2.txt') 