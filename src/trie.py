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
