#!/usr/bin/env python3
"""
Module: 2-build_decision_tree.py

This module implements a simple Decision Tree structure from scratch with:

- Node and Leaf classes representing internal nodes and leaves.
- Recursive computation of maximum depth.
- Recursive counting of nodes, optionally counting only leaves.
- Pretty string representation of the tree with clear formatting.

Classes:
- Node: Represents an internal node with a feature, threshold, children, and depth.
- Leaf: Represents a leaf node holding a prediction value.
- Decision_Tree: Encapsulates the root node and exposes tree-level operations.

The string representation format matches the specified indentation and labeling conventions,
allowing visualization of the tree structure in a human-readable form.
"""

import numpy as np

class Node:
    """
    Class representing an internal node in the decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Threshold value for the split.
        left_child (Node or Leaf): Left subtree or leaf.
        right_child (Node or Leaf): Right subtree or leaf.
        is_leaf (bool): Flag indicating if node is a leaf (False here).
        is_root (bool): Flag indicating if node is the root node.
        depth (int): Depth of this node in the tree (root depth=0).
    """

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None  # Not used currently, placeholder for node data
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively compute the maximum depth in the subtree rooted at this node.

        Returns:
            int: Maximum depth including this node and its descendants.
        """
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively count nodes in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves in the subtree.
        """
        left = self.left_child.count_nodes_below(only_leaves) if self.left_child else 0
        right = self.right_child.count_nodes_below(only_leaves) if self.right_child else 0
        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def __str__(self):
        """
        Generate a string representation of this node and its subtree.

        Returns:
            str: Multiline string representing the subtree.
        """
        if self.is_root:
            header = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            header = f"node [feature={self.feature}, threshold={self.threshold}]"

        result = header
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += "\n" + self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        """
        Add visual prefixes for the left child subtree representation.

        Args:
            text (str): String representation of the left subtree.

        Returns:
            str: Prefixed multiline string with appropriate indentation and branches.
        """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |    " + line + "\n"
        return new_text.rstrip()

    def right_child_add_prefix(self, text):
        """
        Add visual prefixes for the right child subtree representation.

        Args:
            text (str): String representation of the right subtree.

        Returns:
            str: Prefixed multiline string with appropriate indentation and branches.
        """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "         " + line + "\n"
        return new_text.rstrip()


class Leaf(Node):
    """
    Class representing a leaf node in the decision tree.

    Attributes:
        value (any): Prediction or class value stored in the leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf node.

        Returns:
            int: Depth of the leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this leaf (always 1).

        Args:
            only_leaves (bool): Irrelevant here since it's a leaf.

        Returns:
            int: Always 1.
        """
        return 1

    def __str__(self):
        """
        String representation of the leaf node.

        Returns:
            str: Simple string showing leaf value.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Class encapsulating the decision tree structure.

    Attributes:
        root (Node): Root node of the tree.
        max_depth (int): Maximum allowed depth for the tree.
        min_pop (int): Minimum population to allow splits (not used here).
        split_criterion (str): Splitting strategy (not used here).
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
        Compute the maximum depth of the decision tree.

        Returns:
            int: Maximum depth.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the tree, optionally counting only leaves.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        String representation of the entire decision tree.

        Returns:
            str: Multiline string visualizing the tree structure.
        """
        return self.root.__str__()

