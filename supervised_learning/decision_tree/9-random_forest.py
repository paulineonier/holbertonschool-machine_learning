#!/usr/bin/env python3
"""
Decision Tree implementation with support for Gini split criterion.

This script defines a Decision Tree classifier capable of learning a predictive
model from data using either a random or Gini-based splitting strategy.
"""

import numpy as np  # type: ignore


class Node:
    def __init__(self, is_root=False, depth=0, sub_population=None):
        self.is_root = is_root
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.lower = {}
        self.upper = {}
        self.sub_population = sub_population

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf:
    def __init__(self, value):
        self.value = value
        self.depth = None
        self.sub_population = None

    def update_indicator(self):
        pass

    def pred(self, x):
        return self.value


class Decision_Tree:
    def __init__(self, root=None, split_criterion="random", min_pop=5, max_depth=10, seed=0):
        self.root = root if root else Node(is_root=True, depth=0)
        self.split_criterion = split_criterion
        self.min_pop = min_pop
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            values = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(values)
            diff = feature_max - feature_min
        threshold = self.rng.uniform() * (feature_max - feature_min) + feature_min
        return feature, threshold

    def possible_thresholds(self, node, feature):
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        if len(values) <= 1:
            return np.array([values[0]])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        X = self.explanatory[node.sub_population, feature]
        y = self.target[node.sub_population]
        classes = np.unique(self.target)
        n_classes = len(classes)
        thresholds = self.possible_thresholds(node, feature)

        y_one_hot = np.eye(n_classes)[y]
        X_broadcasted = X[:, np.newaxis]
        thresholds_broadcasted = thresholds[np.newaxis, :]

        mask_left = X_broadcasted > thresholds_broadcasted
        mask_right = ~mask_left

        y_left = mask_left[:, :, np.newaxis] * y_one_hot[:, np.newaxis, :]
        y_right = mask_right[:, :, np.newaxis] * y_one_hot[:, np.newaxis, :]

        sum_left = y_left.sum(axis=0)
        sum_right = y_right.sum(axis=0)

        total_left = sum_left.sum(axis=1)
        total_right = sum_right.sum(axis=1)

        gini_left = 1 - np.sum((sum_left / np.maximum(total_left[:, None], 1e-9)) ** 2, axis=1)
        gini_right = 1 - np.sum((sum_right / np.maximum(total_right[:, None], 1e-9)) ** 2, axis=1)

        weights = total_left + total_right
        gini_avg = (total_left * gini_left + total_right * gini_right) / weights

        best_idx = np.argmin(gini_avg)
        return thresholds[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        results = [self.Gini_split_criterion_one_feature(node, i) for i in range(self.explanatory.shape[1])]
        thresholds, impurities = zip(*results)
        best_feature = int(np.argmin(impurities))
        return best_feature, thresholds[best_feature]

    def fit(self, explanatory, target, verbose=0):
        self.explanatory = explanatory
        self.target = target

        if self.split_criterion == "random":
            self._split_criterion_func = self.random_split_criterion
        elif self.split_criterion == "gini":
            self._split_criterion_func = self.Gini_split_criterion
        else:
            raise ValueError(f"Unknown split criterion: {self.split_criterion}")

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
        node.feature, node.threshold = self._split_criterion_func(node)
        feat_vals = self.explanatory[:, node.feature]

        left_population = node.sub_population & (feat_vals > node.threshold)
        right_population = node.sub_population & (feat_vals <= node.threshold)

        def is_leaf(sub_pop):
            if np.sum(sub_pop) == 0:
                return True
            return (
                np.sum(sub_pop) <= self.min_pop or
                node.depth + 1 >= self.max_depth or
                len(np.unique(self.target[sub_pop])) == 1
            )

        if is_leaf(left_population):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_leaf(right_population):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        classes, counts = np.unique(self.target[sub_population], return_counts=True)
        value = classes[np.argmax(counts)]
        leaf = Leaf(value)
        leaf.depth = node.depth + 1
        leaf.sub_population = sub_population
        return leaf

    def get_node_child(self, node, sub_population):
        return Node(depth=node.depth + 1, is_root=False, sub_population=sub_population)

    def update_predict(self):
        self.update_bounds()
        for leaf in self.get_leaves():
            leaf.update_indicator()
        self.predict = lambda X: np.array([self.pred(x) for x in X])

    def pred(self, x):
        return self.root.pred(x)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_leaves(self):
        return self._get_leaves(self.root)

    def _get_leaves(self, node):
        if isinstance(node, Leaf):
            return [node]
        return self._get_leaves(node.left_child) + self._get_leaves(node.right_child)

    def update_bounds(self):
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

        self.root.lower = {i: -100 for i in range(self.explanatory.shape[1])}
        self.root.upper = {i: 100 for i in range(self.explanatory.shape[1])}
        traverse(self.root)

    def depth(self):
        def compute_depth(node):
            if isinstance(node, Leaf):
                return node.depth
            return max(compute_depth(node.left_child), compute_depth(node.right_child))
        return compute_depth(self.root)

    def count_nodes(self, only_leaves=False):
        def count(node):
            if isinstance(node, Leaf):
                return 1 if only_leaves else 0
            return (0 if only_leaves else 1) + count(node.left_child) + count(node.right_child)
        return count(self.root)
