from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import json

from trie.trie import Node

class TCluster:
    """
    Class that implements the T-Cluster algorithm. It uses the DBSCAN algorithm to cluster the points and then
    uses the trie to predict the words that the user is gaze-typing.
    """

    def __init__(self, eps: float=0.1, min_samples: int=5, alpha: float=1, T: float=1, K: int=3, context: list[str]=[], bigram_path: str='data\\bigram.json'):
        """
        Constructor for the TCluster class.
        """

        # DBSCAN parameters
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.model: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples)

        # Sets up the data parameters for later
        self.X: pd.DataFrame = None
        self.labels_: list = None

        # T-Cluster parameters
        self.alpha: float = alpha # Decay factor
        self.T: float = T # Time threshold
        self.K: int = K # Number of keys to consider for each cluster

        # Context parameters for sentences bigrams
        with open(bigram_path, 'r') as f:
            bigram_probs: dict[dict] = json.load(f)
        
        if len(context) < 2:
            last_two: list[str] = ['<s>', '<s>']
        else:
            last_two: list[str] = context[-2:]
        
        self.context_probs: dict[dict] = bigram_probs.get(' '.join(last_two), {})
        



    def fit(self, X: pd.DataFrame, verbose: bool=False):
        """
        Fits the model to the data.

        @params:
        X: pd.DataFrame - The data to fit the model to.
        verbose: bool - Whether to print the labels and core samples.
        """

        self.X = X

        # Fit the DBSCAN model to the data
        self.model.fit(X)

        self.labels_ = self.model.labels_
        self.X['label'] = self.labels_

        self._filter_labels()

        if verbose:
            print(f'{len(self.model.labels_)} Labels:', self.model.labels_)
            print('Core samples:', self.model.core_sample_indices_)

    def _filter_labels(self):
        """
        Filters the labels that are -1 and the labels that are 2 IQRs away from the median size of the clusters.
        """
        
        # Noise points
        self.X = self.X[self.X['label'] != -1]

        # Don't keep the labels that the amount of points are 2 IQRs away from the median
        Q1 = self.X['label'].value_counts().quantile(0.25)
        Q3 = self.X['label'].value_counts().quantile(0.75)
        IQR = Q3 - Q1

        self.X = self.X[self.X['label'].map(self.X['label'].value_counts()) > Q1 - 3 * IQR]

        # Update the labels
        self.labels_ = self.X['label'].tolist()



    def predict(self, keyboard: dict[str, float], trie: Node, verbose: bool=False) -> list:
        """
        Predicts the words that the user is gaze-typing.

        @params:
        keyboard: dict[str, float] - The keyboard layout, with the position of the center and vertices of every key + the letters on each key.
        trie: Node - The trie to predict the words.
        verbose: bool - Whether to print the clusters and the keys that are being pointed to.
        """

        keys = self._find_key_centroid(keyboard, verbose)

        # The hold nodes are the nodes already being considered for words
        hold_nodes = set()

        # Candidate words that the user typed
        candidates = {}

        # For each cluster, update the trie trying to find new candidates
        for key in keys:
            self._update_trie(hold_nodes, candidates, trie, key)

        # Return the top 3 word candidates
        return list(sorted(candidates.items(), key=lambda x: x[1][0], reverse=True))[:3]

    def _find_key_centroid(self, keyboard, verbose: bool=False) -> list:
        """
        Finds the key that the user is pointing to.

        @params:
        keyboard: dict[str, float] - The keyboard layout, with the position of the center and vertices of every key + the letters on each key.
        verbose: bool - Whether to print the clusters and the keys that are being pointed to.
        """
        
        # Calculate the position of the centroid of each cluster
        centroids = self.X.groupby('label')[['x', 'y']].mean()

        keys = []

        # For each cluster, save all distances
        for cluster_id, centroid in centroids.iterrows():
            distances = {}

            # Calculate the distance between the centroid and the center of each key
            for key, points in keyboard.items():
                center, top_left, top_right, bottom_right, bottom_left = points
                distance = np.linalg.norm(np.array(centroid) - np.array(center))
                distances[key] = distance

            # Get the K (hyper-parameter) smallest distances
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            keys.append(sorted_distances[:self.K])

            if verbose:
                print(f"Cluster {cluster_id}: " + ', '.join([f"{key} ({distance:.2f})" for key, distance in sorted_distances[:self.K]]))

        return keys

    def _update_trie(self, hold_nodes: set, candidates: dict, trie: Node, keys: list) -> None:
        """
        Updates the trie with the new keys that the user is pointing to.
        It finds the new nodes based on the keys and updates the candidates,
        trying to form words

        @params:
        hold_nodes: set - The nodes that are already being considered for words.
        candidates: dict - The candidate words that the user is typing.
        trie: Node - The trie to update.
        keys: list - The keys that the user is pointing to.
        """

        # We create a set of new nodes so that the letters from the same key are not considered as children of one another
        new_nodes = set()

        # For each one of the keys
        for val in keys:
            time = 0 #FIXME
            key = val[0] # the string of the key

            # Calculate the score of the key (exponential decay)
            score =  np.exp(-self.alpha*val[1])

            key = key.lower()

            # For every character that is present in that key
            for k in key:
                node = trie.child[ord(k) - ord('a')]

                # If the node exists as start of a word in the vocab, add it to the hold nodes
                # hold nodes being a set guarantees that we don't add the same node twice
                if node is not None:
                    new_nodes.add(node)
                    node.score = max(node.score, score) # update the score of the node

                # Also, get all the current hold nodes and see if their children includes the character that the user is pointing to
                for node in hold_nodes:
                    if node.child[ord(k) - ord('a')] is not None and node.child[ord(k) - ord('a')].letter not in hold_nodes:

                        # Add the child to the new nodes and update the score
                        child = node.child[ord(k) - ord('a')]
                        new_nodes.add(child)
                        child.score = max(child.score, score)
                        
                        # If the child is a word end, add the word to the candidates
                        if child.word_end:
                            for word in child.word:
                                if word not in candidates:
                                    candidates[word] = (self._calculate_candidate_score(child, word), time)

                # Now I need to check the possible permutations of the new nodes
                permutation_nodes = set()

                for node in new_nodes:
                    for other in new_nodes:
                        if node.letter != other.letter and node.child[ord(other.letter) - ord('a')] is not None:

                            # Add the child to the new nodes and update the score
                            child = node.child[ord(other.letter) - ord('a')]
                            permutation_nodes.add(child)
                            child.score = max(child.score, score)
                            
                            if child.word_end:
                                for word in child.word:
                                    if word not in candidates:
                                        candidates[word] = (self._calculate_candidate_score(child, word), time)

                # Merge the permutation nodes with the new nodes
                new_nodes.update(permutation_nodes)
                                    

        # Merge the hold nodes with the new nodes
        hold_nodes.update(new_nodes)

        self._update_candidates(candidates, time)


    def _calculate_candidate_score(self, node: Node, word: str) -> float:
        """
        Calculates the score of a candidate word.

        @params:
        node: Node - The node that is the end of the candidate word.
        """
        
        # Goes up the trie and sums the score of the nodes
        score = 0
        while node.parent is not None:
            score += node.score
            node = node.parent

        return score * self._linguistic_score(word)
    
    def _linguistic_score(self, word: str) -> float:
        """
        Calculates the linguistic score of a word based on a bigram model.

        @params:
        word: str - The word to calculate the linguistic score.
        """
        
        # Get the vocabulary size (V) as the number of unique words in the bigram model
        V: int = len(self.context_probs)

        # Retrieve the probability of the word or assign 0 if it's not found
        word_prob: float = self.context_probs.get(word, 0)

        # Apply smoothing
        smoothed_prob: float = (word_prob + self.alpha) / (sum(self.context_probs.values()) + self.alpha * (V + 1))

        return smoothed_prob

    def _update_candidates(self, candidates: dict, time: float) -> None:
        """
        Checks the candidates that are being considered as words if they need to be removed.

        @params:
        candidates: dict - The candidates that are being considered as words.
        time: float - The time that has passed since the last update.
        """

        # Remove the candidates that are there for more than T time
        for key in list(candidates.keys()):
            if time - candidates[key][1] > self.T:
                del candidates[key]
