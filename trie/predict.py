from trie.trie import Node
from trie.keyboard import find_key
from trie.util import update_trie, update_hold_nodes
from trie.score import score
from trie.candidates import score_candidates


def predict(coordinates: list[tuple], keyboard: dict, root: Node, word: str = 'TEST', verbose=False) -> list[str]:
    '''
    For a given list of eye coordinates over time, predict the most likely words that the user is typing

    Args:
    - word: str - the word that the user is typing
    - coordinates: list[tuple] - a list of tuples with the x, y coordinates of the eye while typing the word
    - keyboard: dict - a dictionary with the keyboard layout
    - root: Node - the root of the trie
    '''

    hold_nodes = set()
    candidates = dict()
    prev = None

    for i in range(len(coordinates)):
        # Find the key that the user is pointing to
        key = find_key(coordinates[i], keyboard)

        #print(f"{coordinates[i]} {key}")

        if key is not None:
            # Get the last 20 points
            idx = i - 20 if i > 20 else 0
            key_score = score(coordinates[idx:i], coordinates[i], keyboard[key][0])

            # If the key is different from the previous one, update the trie
            if key != prev:
                update_trie(hold_nodes, candidates, root, key, key_score, coordinates[i][2])
            # Otherwise, update only the hold nodes
            else:
                update_hold_nodes(hold_nodes, key, key_score, candidates, coordinates[i][2])
            
        prev = key
    
    results = find_best_candidates(candidates)

    if verbose:
        print(word)
        print(results)

    return results


def find_best_candidates(candidates: dict):
    '''
    Find the best candidates for the word that the user is typing

    Args:
    - candidates: dict - a dictionary with the candidates and their scores

    Returns:
    - list[str] - a list with the best candidates
    '''

    scores = score_candidates(candidates)

    return list(sorted(scores, key=scores.get, reverse=True))[:3]
