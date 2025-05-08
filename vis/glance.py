import numpy as np
from collections import defaultdict, deque
import math
from sklearn.cluster import DBSCAN

class TrieNode:
    def __init__(self, char=''):
        self.char = char
        self.children = {}
        self.word = None
        self.key_score = 0
        self.sum_score = 0
        self.state = 'RELEASE'  # or 'HOLD'

class GlanceWriterDecoder:
    def __init__(self, key_positions, dictionary, sigma=0.4, stability_window=30, eps=0.07, min_samples=5):
        """
        key_positions: dict of {key: [center, top_left, top_right, bottom_right, bottom_left]} 
                      where each point is (x, y)
        dictionary: list of valid words
        sigma: std deviation for distance score
        stability_window: window size in pixels for stability score
        eps: DBSCAN clustering parameter
        min_samples: DBSCAN minimum samples parameter
        """
        self.sigma = sigma
        self.stability_window = stability_window
        self.key_positions = key_positions
        self.root = TrieNode()
        self.build_trie(dictionary)
        self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)

    def build_trie(self, words):
        for word in words:
            node = self.root
            i = 0
            while i < len(word):
                ch = word[i]
                while i + 1 < len(word) and word[i + 1] == ch:
                    i += 1  # merge repeated chars
                if ch not in node.children:
                    node.children[ch] = TrieNode(ch)
                node = node.children[ch]
                i += 1
            node.word = word

    def gaussian_score(self, distance):
        return (1 / (self.sigma * math.sqrt(2 * math.pi))) * math.exp(-distance**2 / (2 * self.sigma**2))

    def compute_stability(self, points, i):
        speeds = []
        for j in range(max(0, i - self.stability_window), i):
            dt = points[j+1][2] - points[j][2]
            if dt == 0: continue
            dx = points[j+1][0] - points[j][0]
            dy = points[j+1][1] - points[j][1]
            speeds.append(math.sqrt(dx**2 + dy**2) / dt)
        if not speeds:
            return 0
        return 1 / (np.mean(speeds) + 1e-5)  # Avoid division by zero

    def decode(self, gaze_points, top_k=5):
        """
        gaze_points: list of (x, y, time) tuples
        top_k: number of top predictions to return
        """
        # Convert gaze points to numpy array for clustering
        points_array = np.array(gaze_points)
        
        # Cluster the points
        labels = self.cluster_model.fit_predict(points_array[:, :2])
        
        # Group points by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Skip noise points
                clusters[label].append(gaze_points[i])
        
        hold_nodes = set()
        candidates = {}

        # Process each cluster
        for cluster_id, cluster_points in clusters.items():
            # Calculate cluster centroid
            centroid = np.mean([(p[0], p[1]) for p in cluster_points], axis=0)
            
            # Find closest keys to centroid
            key_scores = []
            for key, key_points in self.key_positions.items():
                center = key_points[0]  # Use center point of key
                distance = np.linalg.norm(np.array(centroid) - np.array(center))
                score = self.gaussian_score(distance)
                key_scores.append((key, score))
            
            # Sort keys by score and take top K
            key_scores.sort(key=lambda x: -x[1])
            top_keys = key_scores[:3]  # Use top 3 keys per cluster
            
            # Update trie with these keys
            for key, score in top_keys:
                for ch in key.lower():
                    if ch not in self.root.children:
                        continue
                        
                    nodes_to_update = deque([(self.root.children[ch], score)])
                    while nodes_to_update:
                        node, key_score = nodes_to_update.popleft()
                        if node.state == 'RELEASE':
                            node.state = 'HOLD'
                        if key_score > node.key_score:
                            node.key_score = key_score
                            # Update sum to root
                            temp = node
                            total = 0
                            while temp:
                                total += temp.key_score
                                temp = self._get_parent(temp)
                            node.sum_score = total

                        for child in node.children.values():
                            nodes_to_update.append((child, key_score))

        # Collect candidates and sort by score
        self.collect_candidates(self.root, '', candidates)
        sorted_words = sorted(candidates.items(), key=lambda x: -x[1])
        return [word for word, _ in sorted_words[:top_k]]

    def collect_candidates(self, node, path, candidates):
        if node.word:
            candidates[node.word] = node.sum_score
        for child in node.children.values():
            self.collect_candidates(child, path + child.char, candidates)

    def _get_parent(self, node):
        # Optional helper to retrieve parent for scoring; implement as needed
        return None



