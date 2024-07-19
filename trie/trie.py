import pandas as pd

class Node:
    def __init__(self, letter=None):
        self.word_end = False
        self.word = []
        self.letter = letter

        self.parent = None
        self.child = [None]*26
        self.score = 0

    def __repr__(self):
        return f"<Node | Letter {self.letter} | Score {self.score} | Words [{self.word}] >"


def create_trie(path: str) -> Node:
    root = Node()

    if path.endswith('.xlsx'):
        df = pd.read_excel(path, sheet_name='4 forms (219k)')
        training_words = df['word'].tolist()
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        training_words = df['word'].tolist()
    else:
        with open(path, 'r') as f:
            training_words = f.readlines()
    
    training_words = [str(word).lower() for word in training_words if str(word).isalpha()]

    for word in training_words:
        insert_key(root, word)

    return root


def insert_key(root: Node, key: str):
    curr = root

    for i in range(len(key)):
        if i < len(key) - 1 and key[i] == key[i + 1]:
            continue

        char = key[i]

        if curr.child[ord(char) - ord('a')] is None:
            new_node = Node(char)
            new_node.parent = curr
            curr.child[ord(char) - ord('a')] = new_node

        curr = curr.child[ord(char) - ord('a')]

    curr.word_end = True
    curr.word.append(key)


def search_trie(root: Node, key: str):
    curr = root

    for i in range(len(key)):
        char = key[i]

        if curr.child[ord(char) - ord('a')] is None:
            return False

        curr = curr.child[ord(char) - ord('a')]

    return curr.word_end and key in curr.word


def print_trie(root: Node):
    print(root)
    for c in root.child:
        if c is not None: print_trie(c)


def main():
    # Testing with simple words
    root = Node()

    insert_key(root, 'more')
    insert_key(root, 'moore')

    print_trie(root)


if __name__ == "__main__":
    main()
