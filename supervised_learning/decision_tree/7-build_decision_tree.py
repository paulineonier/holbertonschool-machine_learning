#!/usr/bin/env python3
"""
Decision Tree implementation with random splitting criterion.

This script defines a trainable decision tree capable of classifying data based
on recursive binary splits. The splitting strategy is either random or (optionally)
Gini-based. Each node in the tree keeps track of which samples reach it.

Author: OpenAI
"""

import numpy as np

class Node:
    """
    Represents an internal decision node in the tree.
    """
    def __init__(self, is_root=False, depth=0):
        self.is_root = is_root
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.lower = {}
        self.upper = {}
        self.sub_population = None  # Boolean mask of samples that reach this node

    def pred(self, x):
        """
        Route the input sample x through the tree starting from this node.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf:
    """
    Represents a terminal leaf node that makes a class prediction.
    """
    def __init__(self, value):
        self.value = value
        self.depth = None
        self.sub_population = None

    def update_indicator(self):
        """
        Optional placeholder for post-training analysis or visualization.
        """
        pass

    def pred(self, x):
        """
        Return the predicted class stored in this leaf.
        """
        return self.value


class Decision_Tree:
    """
    Main Decision Tree class capable of training and making predictions.
    """
    def __init__(self, root=None, split_criterion="random", min_pop=5, max_depth=10, seed=0):
        """
        Initialize the tree with optional root node and parameters.
        """
        self.root = root if root else Node(is_root=True, depth=0)
        self.split_criterion = split_criterion
        self.min_pop = min_pop
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def np_extrema(self, arr):
        """
        Return the min and max of a numpy array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Generate a random feature and threshold to split the node.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            values = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(values)
            diff = feature_max - feature_min
        threshold = self.rng.uniform() * (feature_max - feature_min) + feature_min
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Train the decision tree on the given dataset.
        """
        self.explanatory = explanatory
        self.target = target

        # Assign the splitting function
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion  # Placeholder

        # Start with the entire dataset at the root
        self.root.sub_population = np.ones_like(target, dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(explanatory, target)}""")

    def fit_node(self, node):
        """
        Recursively build the decision tree from the current node.
        """
        node.feature, node.threshold = self.split_criterion(node)
        feat_vals = self.explanatory[:, node.feature]

        # Boolean masks for left and right child nodes
        left_population = node.sub_population & (feat_vals > node.threshold)
        right_population = node.sub_population & (feat_vals <= node.threshold)

        # Determine whether a node should be a leaf
        def is_leaf(sub_pop):
            return (
                np.sum(sub_pop) < self.min_pop or
                node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[sub_pop])) == 1
            )

        # Process left child
        if is_leaf(left_population):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Process right child
        if is_leaf(right_population):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create and return a leaf node for the given sub-population.
        """
        classes, counts = np.unique(self.target[sub_population], return_counts=True)
        value = classes[np.argmax(counts)]  # Most frequent class
        leaf = Leaf(value)
        leaf.depth = node.depth + 1
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        """
        Create and return a new internal node for the given sub-population.
        """
        return Node(depth=node.depth + 1, is_root=False, sub_population=sub_population)

    def update_predict(self):
        """
        Update the prediction function after training.
        """
        self.update_bounds()
        for leaf in self.get_leaves():
            leaf.update_indicator()
        self.predict = lambda X: np.array([self.pred(x) for x in X])

    def pred(self, x):
        """
        Predict a single instance x by routing through the tree.
        """
        return self.root.pred(x)

    def accuracy(self, X, y):
        """
        Compute accuracy on given dataset.
        """
        return np.mean(self.predict(X) == y)

    def get_leaves(self):
        """
        Return a list of all leaf nodes in the tree.
        """
        return self._get_leaves(self.root)

    def _get_leaves(self, node):
        if isinstance(node, Leaf):
            return [node]
        return self._get_leaves(node.left_child) + self._get_leaves(node.right_child)

    def update_bounds(self):
        """
        Optional: update lower and upper bounds per node for visualization.
        """
        def traverse(node):
            if isinstance(node, Leaf):
                return
            f = node.feature
            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.lower[f] = node.threshold
            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.upper[f] = node.threshold
            traverse(node.left_child)
            traverse(node.right_child)

        # Initialize bounds at root
        self.root.lower = {i: -100 for i in range(self.explanatory.shape[1])}
        self.root.upper = {i: 100 for i in range(self.explanatory.shape[1])}
        traverse(self.root)

    def depth(self):
        """
        Compute and return the depth of the tree.
        """
        def compute_depth(node):
            if isinstance(node, Leaf):
                return node.depth
            return max(compute_depth(node.left_child), compute_depth(node.right_child))
        return compute_depth(self.root)

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree. Optionally count only leaves.
        """
        def count(node):
            if isinstance(node, Leaf):
                return 1 if only_leaves else 0
            return (0 if only_leaves else 1) + count(node.left_child) + count(node.right_child)
        return count(self.root)