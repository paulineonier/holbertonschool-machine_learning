#!/usr/bin/env python3
"""
This module defines a decision tree with structure, traversal utilities, 
depth calculation, node counting, and a pretty-printed string output.
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
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively computes the maximum depth among all descendant nodes.
        """
        if self.is_leaf:
            return self.depth
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts the number of nodes in the subtree rooted at this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves below this node (inclusive).
        """
        left = self.left_child.count_nodes_below(only_leaves) if self.left_child else 0
        right = self.right_child.count_nodes_below(only_leaves) if self.right_child else 0
        return left + right if only_leaves else 1 + left + right

    def __str__(self):
        """
        Pretty string representation of the node and its children, recursively.

        Returns:
            str: Tree structure starting from this node.
        """
        text = "[X{} < {}]".format(self.feature, self.threshold) if self.is_root else "-> node [X{} < {}]".format(self.feature, self.threshold)

        if self.left_child:
            left_text = self.left_child.__str__()
            text += "\n" + self.left_child_add_prefix(left_text)

        if self.right_child:
            right_text = self.right_child.__str__()
            text += "\n" + self.right_child_add_prefix(right_text)

        return text

    def left_child_add_prefix(self, text):
        """
        Adds the visual prefix for the left child subtree.

        Args:
            text (str): String representation of left child.

        Returns:
            str: Prefixed string for display.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |  " + line + "\n"
        return new_text.rstrip()

    def right_child_add_prefix(self, text):
        """
        Adds the visual prefix for the right child subtree.

        Args:
            text (str): String representation of right child.

        Returns:
            str: Prefixed string for display.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "       " + line + "\n"
        return new_text.rstrip()

class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

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
        """
        Returns the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes or leaves in the tree.

        Args:
            only_leaves (bool): Whether to count only the leaf nodes.

        Returns:
            int: Number of relevant nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the full tree.
        """
        return self.root.__str__()
