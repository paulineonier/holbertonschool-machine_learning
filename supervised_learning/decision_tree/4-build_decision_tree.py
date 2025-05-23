#!/usr/bin/env python3
"""
Module: 4-build_decision_tree.py

Adds bound propagation for each feature in the decision tree.
Each node computes the bounds (min/max) of the feature values it splits on.
"""

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        left = self.left_child.count_nodes_below(only_leaves) if self.left_child else 0
        right = self.right_child.count_nodes_below(only_leaves) if self.right_child else 0
        return (0 if only_leaves else 1) + left + right

    def get_leaves_below(self):
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        # Initial bounds for root
        if self.is_root:
            self.lower = {self.feature: -np.inf}
            self.upper = {self.feature: np.inf}

        # Propagate bounds to children
        for child, direction in zip([self.left_child, self.right_child], ['left', 'right']):
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                if direction == 'left':
                    child.upper[self.feature] = min(child.upper.get(self.feature, np.inf), self.threshold)
                else:
                    child.lower[self.feature] = max(child.lower.get(self.feature, -np.inf), self.threshold)

        # Recursively update children
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def __str__(self):
        header = (
            f"root [feature={self.feature}, threshold={self.threshold}]"
            if self.is_root else
            f"node [feature={self.feature}, threshold={self.threshold}]"
        )
        result = header
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += "\n" + self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        return "    +---> " + lines[0] + "\n" + "".join("    |    " + l + "\n" for l in lines[1:]).rstrip()

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        return "    +---> " + lines[0] + "\n" + "".join("         " + l + "\n" for l in lines[1:]).rstrip()


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        # Leaves do not propagate bounds further
        pass

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def __str__(self):
        return self.root.__str__()