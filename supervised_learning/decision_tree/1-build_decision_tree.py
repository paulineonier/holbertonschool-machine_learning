#!/usr/bin/env python3
"""
This module defines a simple implementation of a decision tree from scratch.

It includes:
- Node: internal decision node
- Leaf: terminal output node
- Decision_Tree: manages the tree and provides helper methods

The tree supports:
- Depth computation
- Node/leaf counting
"""

import numpy as np

class Node:
    """
    Represents an internal decision node in the tree.

    Attributes:
        feature (int): Index of the feature to split on.
        threshold (float): Value to split the feature on.
        left_child (Node or Leaf): Left child node.
        right_child (Node or Leaf): Right child node.
        is_leaf (bool): Whether this node is a leaf.
        is_root (bool): Whether this node is the root.
        sub_population (any): Placeholder for data tracking.
        depth (int): Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively computes the maximum depth in the subtree rooted at this node.

        Returns:
            int: Maximum depth found in subtree.
        """
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts the number of nodes (or only leaves) in the subtree.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of matching nodes.
        """
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves) if self.left_child else 0
        right_count = self.right_child.count_nodes_below(only_leaves=only_leaves) if self.right_child else 0

        if only_leaves:
            return left_count + right_count
        else:
            return 1 + left_count + right_count


class Leaf(Node):
    """
    Represents a leaf node, which holds a prediction or classification result.

    Attributes:
        value (any): The output value at this leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns:
            int: The depth of the leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns:
            int: Always 1, as a leaf is a single node.
        """
        return 1


class Decision_Tree:
    """
    Represents the decision tree and provides utility methods.

    Attributes:
        max_depth (int): Max allowed tree depth.
        min_pop (int): Minimum population (samples) required to split a node.
        seed (int): Random seed for reproducibility.
        split_criterion (str): Criterion used to split nodes.
        root (Node): Root node of the tree.
        explanatory (np.ndarray): Input feature matrix.
        target (np.ndarray): Target values.
        predict (callable): Prediction function (to be implemented).
    """

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
        """
        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count all nodes or only leaves in the tree.

        Args:
            only_leaves (bool): If True, counts only leaves.

        Returns:
            int: Total number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
 