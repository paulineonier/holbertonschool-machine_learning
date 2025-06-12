#!/usr/bin/env python3

Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree
import numpy as np

class Isolation_Random_Forest:
    """
    Isolation Random Forest for anomaly detection.

    This class implements an ensemble of Isolation Random Trees used to compute
    anomaly scores based on the average path length (depth) of samples in the trees.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth allowed for each tree.
        seed (int): Random seed for reproducibility.
        numpy_preds (list): List of prediction functions from each tree.
        explanatory (np.ndarray): Training data features.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initialize the Isolation Random Forest.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            min_pop (int): Minimum population for splitting a node (not used directly here).
            seed (int): Seed for random number generator.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Compute the average depth of samples in the forest.

        Args:
            explanatory (np.ndarray): Samples to predict depths for.

        Returns:
            np.ndarray: Average depths for each sample.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Train the Isolation Random Forest by fitting n_trees Isolation Random Trees.

        Args:
            explanatory (np.ndarray): Training data features.
            n_trees (int): Number of trees to fit.
            verbose (int): Verbosity level (0 = silent, 1 = print info).
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth, seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Identify the n_suspects samples with the smallest average depth (most anomalous).

        Args:
            explanatory (np.ndarray): Data samples to evaluate.
            n_suspects (int): Number of suspect samples to return.

        Returns:
            tuple: (suspect_samples, suspect_depths)
                suspect_samples (np.ndarray): Samples identified as suspects.
                suspect_depths (np.ndarray): Corresponding average depths of these samples.
        """
        depths = self.predict(explanatory)
        suspects_idx = np.argsort(depths)[:n_suspects]
        return explanatory[suspects_idx], depths[suspects_idx]
