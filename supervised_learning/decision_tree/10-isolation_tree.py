#!/usr/bin/env python3
"""
Isolation Random Tree for outlier detection.

This class builds a random tree that isolates samples by recursively splitting
data with random thresholds on random features, until max depth or minimal
population size is reached. The prediction for a sample is the depth of the leaf
where it falls, smaller depths indicating potential outliers.
"""

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
import numpy as np


class Isolation_Random_Tree:
    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True, depth=0)
        self.explanatory = None
        self.max_depth = max_depth
        self.min_pop = 1
        self.predict = None

    def __str__(self):
        # same as in Decision_Tree
        def recurse(node, depth=0):
            prefix = "  " * depth
            if isinstance(node, Leaf):
                return f"{prefix}Leaf(depth={node.depth}, size={np.sum(node.sub_population)})\n"
            else:
                s = f"{prefix}Node(depth={node.depth}, feature={node.feature}, threshold={node.threshold:.3f})\n"
                s += recurse(node.left_child, depth + 1)
                s += recurse(node.right_child, depth + 1)
                return s
        return recurse(self.root)

    def depth(self):
        # same as in Decision_Tree
        def compute_depth(node):
            if isinstance(node, Leaf):
                return node.depth
            return max(compute_depth(node.left_child), compute_depth(node.right_child))
        return compute_depth(self.root)

    def count_nodes(self, only_leaves=False):
        # same as in Decision_Tree
        def count(node):
            if isinstance(node, Leaf):
                return 1 if only_leaves else 0
            return (0 if only_leaves else 1) + count(node.left_child) + count(node.right_child)
        return count(self.root)

    def update_bounds(self):
        # same as in Decision_Tree
        def traverse(node):
            if isinstance(node, Leaf):
                return
            f = node.feature
            node.left_child.lower = node.lower.copy()
            node.left_child.upper = node.upper.copy()
            node.left_child.upper[f] = node.threshold
            node.right_child.lower = node.lower.copy()
            node.right_child.upper = node.upper.copy()
            node.right_child.lower[f] = node.threshold
            traverse(node.left_child)
            traverse(node.right_child)

        self.root.lower = {i: -np.inf for i in range(self.explanatory.shape[1])}
        self.root.upper = {i: np.inf for i in range(self.explanatory.shape[1])}
        traverse(self.root)

    def get_leaves(self):
        # same as in Decision_Tree
        def collect_leaves(node):
            if isinstance(node, Leaf):
                return [node]
            return collect_leaves(node.left_child) + collect_leaves(node.right_child)
        return collect_leaves(self.root)

    def update_predict(self):
        # same as in Decision_Tree
        self.update_bounds()
        for leaf in self.get_leaves():
            # no indicator update needed, just pass
            pass
        self.predict = lambda X: np.array([self.pred(x) for x in X])

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        # same as in Decision_Tree
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            values = self.explanatory[:, feature][node.sub_population]
            f_min, f_max = self.np_extrema(values)
            diff = f_max - f_min
        threshold = self.rng.uniform() * (f_max - f_min) + f_min
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        leaf = Leaf(value=None)  # no value needed for isolation tree
        leaf.depth = node.depth + 1
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        return Node(depth=node.depth + 1, is_root=False, sub_population=sub_population)

    def fit_node(self, node):
        node.feature, node.threshold = self.random_split_criterion(node)
        feat_vals = self.explanatory[:, node.feature]

        left_population = node.sub_population & (feat_vals <= node.threshold)
        right_population = node.sub_population & (feat_vals > node.threshold)

        is_left_leaf = (np.sum(left_population) <= self.min_pop) or (node.depth + 1 >= self.max_depth)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (np.sum(right_population) <= self.min_pop) or (node.depth + 1 >= self.max_depth)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")

    def pred(self, x):
        # prediction is the depth of the leaf node where x falls
        node = self.root
        while not isinstance(node, Leaf):
            if x[node.feature] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.depth
