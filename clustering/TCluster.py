from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

from trie.trie import Node

class TCluster:
    def __init__(self, eps: float=0.1, min_samples: int=5, alpha: float=1, T=1, K=3):
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.model: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples)

        self.X: pd.DataFrame = None
        self.labels_: list = None

        self.alpha = alpha
        self.T = T
        self.K = K
    
    def fit(self, X: pd.DataFrame, verbose: bool=False):
        self.X = X
        self.model.fit(X)

        self.labels_ = self.model.labels_
        self.X['label'] = self.labels_

        self.filter_labels()

        if verbose:
            print(f'{len(self.model.labels_)} Labels:', self.model.labels_)
            print('Core samples:', self.model.core_sample_indices_)
    
    def filter_labels(self):
        self.X = self.X[self.X['label'] != -1]

        # Don't keep the labels that the amount of points are 2 IQRs away from the median
        Q1 = self.X['label'].value_counts().quantile(0.25)
        Q3 = self.X['label'].value_counts().quantile(0.75)
        IQR = Q3 - Q1

        self.X = self.X[self.X['label'].map(self.X['label'].value_counts()) > Q1 - 3 * IQR]
        
        self.labels_ = self.X['label'].tolist()



    def predict(self, keyboard: dict[str, float], trie: Node, verbose: bool=False) -> list:
        keys = self.find_key_centroid(keyboard, verbose)

        hold_nodes = set()
        candidates = {}

        for key in keys:
            self.update_trie(hold_nodes, candidates, trie, key)
        
        return list(sorted(candidates.items(), key=lambda x: x[1][0], reverse=True))[:3]


    def find_key_centroid(self, keyboard, verbose: bool=False) -> list:
        centroids = self.X.groupby('label')[['x', 'y']].mean()

        keys = []
        
        for cluster_id, centroid in centroids.iterrows():
            distances = {}
            for key, points in keyboard.items():
                center, top_left, top_right, bottom_right, bottom_left = points
                distance = np.linalg.norm(np.array(centroid) - np.array(center))
                distances[key] = distance
            
            # K smallest distances
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])

            keys.append(sorted_distances[:self.K])

            if verbose:
                print(f"Cluster {cluster_id}: " + ', '.join([f"{key} ({distance:.2f})" for key, distance in sorted_distances[:self.K]]))
        
        return keys
    
    def update_trie(self, hold_nodes: set, candidates: dict, trie: Node, keys: list):
        
        new_nodes = set()

        for val in keys:
            time = 0 #FIXME
            key = val[0]
            score =  np.exp(-self.alpha*val[1])

            # If the user is pointing to a key, get the initial node for that letter and put it in the hold set if it's not there
            key = key.lower()

            # For now, iterate all the caracters of the keyboard
            for k in key:
                node = trie.child[ord(k) - ord('a')]

                # Update the node that starts new words
                if node is not None:
                    hold_nodes.add(node)

                # Also, get all the hold nodes and see if their children are the character that the user is pointing to
                for node in hold_nodes:
                    if node.child[ord(k) - ord('a')] is not None and node.child[ord(k) - ord('a')].letter not in hold_nodes:
                        child = node.child[ord(k) - ord('a')]
                        new_nodes.add(child)
                        child.score = max(child.score, score)

                        if child.word_end:
                            for word in child.word:
                                if word not in candidates:
                                    candidates[word] = (self.calculate_candidate_score(child), time)
                
        # Merge the hold nodes with the new nodes
        hold_nodes.update(new_nodes)

        self.update_candidates(candidates, time)


    def calculate_candidate_score(self, node: Node) -> float:
        score = 0
        while node.parent is not None:
            score += node.score
            node = node.parent
        
        return score


    def update_candidates(self, candidates: dict, time: float) -> None:
        # Remove the candidates that are there for more than T time
        for key in list(candidates.keys()):
            if time - candidates[key][1] > self.T:
                del candidates[key]