#!/usr/bin/env python3
"""
Module: 3-build_decision_tree.py

This module defines a simple implementation of a decision tree, composed of Node and Leaf classes,
and a Decision_Tree container. It includes methods for displaying the tree and retrieving all leaves.

Classes:
- Node: Represents internal decision nodes.
- Leaf: Represents terminal nodes storing predicted values.
- Decision_Tree: Wraps the tree and exposes operations on it.
"""

import numpy as np

class Node:
    """
    Class representing an internal decision node in a decision tree.

    Attributes:
        feature (int): Feature index used for the split.
        threshold (float): Threshold value used for the split.
        left_child (Node or Leaf): Left child subtree.
        right_child (Node or Leaf): Right child subtree.
        is_leaf (bool): Always False for Node.
        is_root (bool): Whether this node is the root of the tree.
        depth (int): Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def max_depth_below(self):
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        left = self.left_child.count_nodes_below(only_leaves) if self.left_child else 0
        right = self.right_child.count_nodes_below(only_leaves) if self.right_child else 0
        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def get_leaves_below(self):
        """
        Recursively collect all leaf nodes under this node.

        Returns:
            list of Leaf: All leaves in the subtree rooted at this node.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

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
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |    " + line + "\n"
        return new_text.rstrip()

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "         " + line + "\n"
        return new_text.rstrip()


class Leaf(Node):
    """
    Class representing a leaf (terminal node) in the decision tree.

    Attributes:
        value (any): Predicted value stored in the leaf.
        depth (int): Depth in the tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        """
        Return this leaf as a list.

        Returns:
            list of Leaf: A single-element list containing self.
        """
        return [self]

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Class encapsulating the decision tree.

    Attributes:
        root (Node): The root node of the tree.
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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Return all leaf nodes in the tree.

        Returns:
            list of Leaf: All leaves from the tree.
        """
        return self.root.get_leaves_below()

    def __str__(self):
        return self.root.__str__()
