from trie.trie import Node

T = 1

def update_trie(hold_nodes: set, candidates: dict, trie: Node, key: str, score: float, time: float):
    # If the user is pointing to a key, get the initial node for that letter and put it in the hold set if it's not there
    key = key.lower()

    node = trie.child[ord(key) - ord('a')]

    # Update the node that starts new words
    if node is not None:
        hold_nodes.add(node)
    
    new_nodes = set()

    # Also, get all the hold nodes and see if their children are the character that the user is pointing to
    for node in hold_nodes:
        if node.letter == key:
            node.score = max(node.score, score)

        if node.child[ord(key) - ord('a')] is not None and node.child[ord(key) - ord('a')].letter not in hold_nodes:
            child = node.child[ord(key) - ord('a')]
            new_nodes.add(child)
            child.score = max(child.score, score)

            if child.word_end:
                for word in child.word:
                    if word not in candidates:
                        candidates[word] = (calculate_candidate_score(child), time)
    
    # Merge the hold nodes with the new nodes
    hold_nodes.update(new_nodes)

    update_candidates(candidates, time)


def calculate_candidate_score(node: Node) -> float:
    score = 0
    while node.parent is not None:
        score += node.score
        node = node.parent
    
    return score

def update_hold_nodes(hold_nodes: set, key: str, score: float, candidates: dict, time: float):
    key = key.lower()

    for node in hold_nodes:
        if node.letter == key:
            node.score = max(node.score, score)
    
    update_candidates(candidates, time)

def update_candidates(candidates: dict, time: float):
    # Remove the candidates that are there for more than T time
    for key in list(candidates.keys()):
        if time - candidates[key][1] > T:
            del candidates[key]