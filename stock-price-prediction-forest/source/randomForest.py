from decisionTree import DecisionTree
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_sample_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_sample_split = min_sample_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        self.oob_preds = np.zeros(n_samples, dtype=int)
        self.oob_counts = np.zeros(n_samples, dtype=int)
        
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,  
                n_features=self.n_features,  
                min_samples_split=self.min_sample_split
            )
            X_sample, y_sample, oob_idx = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # Collect OOB predictions
            oob_preds, oob_counts = self._update_oob_predictions(X, tree, oob_idx)

        # Calculate OOB error
        self.oob_error = np.mean(self.oob_preds != y)
        print(f"OOB Error: {self.oob_error:.4f}")

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(idx))
        return X[idx], y[idx], oob_idx

    def _update_oob_predictions(self, X, tree, oob_idx):
        oob_preds = np.zeros(len(oob_idx), dtype=int)
        oob_counts = np.zeros(len(oob_idx), dtype=int)
        
        for i in oob_idx:
            pred = tree.predict(X[i].reshape(1, -1))[0]
            self.oob_preds[i] += pred
            self.oob_counts[i] += 1
        
        return self.oob_preds, self.oob_counts

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

    def plot_oob_error(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.oob_error, marker='o', linestyle='-')
        plt.title('Out-Of-Bag Error Rate')
        plt.xlabel('Number of Trees')
        plt.ylabel('OOB Error Rate')
        plt.grid(True)
        plt.show()
