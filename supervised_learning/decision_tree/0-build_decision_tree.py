#!/usr/bin/env python3
"""
This module defines classes for building a decision tree from scratch.
It includes Node, Leaf, and Decision_Tree classes, and supports recursive
computation of the maximum depth of a decision tree.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Threshold value for splitting.
        left_child (Node or Leaf): Left child node.
        right_child (Node or Leaf): Right child node.
        is_leaf (bool): Flag indicating if the node is a leaf.
        is_root (bool): Flag indicating if the node is the root.
        sub_population (any): Placeholder for data associated with the node.
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
        Recursively compute the maximum depth of the subtree below this node.

        Returns:
            int: Maximum depth among all descendant nodes.
        """
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth

        return max(left_depth, right_depth)


class Leaf(Node):
    """
    Represents a leaf node in a decision tree. Inherits from Node.

    Attributes:
        value (any): The predicted value or label at this leaf.
        depth (int): Depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf node.

        Returns:
            int: Depth of the current leaf node.
        """
        return self.depth


class Decision_Tree:
    """
    Represents a decision tree classifier or regressor.

    Attributes:
        max_depth (int): Maximum allowed depth of the tree.
        min_pop (int): Minimum population required to split.
        seed (int): Random seed for reproducibility.
        split_criterion (str): Splitting strategy ("random", etc.).
        root (Node): Root node of the tree.
        explanatory (np.ndarray): Placeholder for input features.
        target (np.ndarray): Placeholder for target values.
        predict (callable): Prediction function (to be implemented).
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0, 
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Compute the total depth of the decision tree.

        Returns:
            int: Maximum depth of the tree from the root.
        """
        return self.root.max_depth_below()