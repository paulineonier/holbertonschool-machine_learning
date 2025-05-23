#!/usr/bin/env python3
import numpy as np

class Node:
    """
    Represents a decision node in a decision tree.

    Attributes:
        feature (int): Feature index on which the node splits.
        threshold (float): Threshold value for the split.
        left_child (Node or Leaf): Left subtree.
        right_child (Node or Leaf): Right subtree.
        depth (int): Depth of the node in the tree.
        is_root (bool): True if the node is the root of the tree.
        is_leaf (bool): False for Node objects.
    """
    def __init__(self, feature, threshold, left_child, right_child, depth, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def get_leaves_below(self):
        """
        Recursively returns a list of all Leaf nodes below this Node.
        """
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Computes lower and upper bounds for features and propagates them recursively to child nodes.
        """
        if self.is_root:
            self.lower = {self.feature: -np.inf}
            self.upper = {self.feature: np.inf}

        for child, is_left in zip([self.left_child, self.right_child], [True, False]):
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if is_left:
                child.upper[self.feature] = min(child.upper.get(self.feature, np.inf), self.threshold)
            else:
                child.lower[self.feature] = max(child.lower.get(self.feature, -np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Builds a lambda function that, for a given 2D NumPy array x,
        returns a boolean array indicating which samples fall under this node's conditions.
        """
        def is_large_enough(x):
            return np.all(np.array([x[:, k] > self.lower[k] for k in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, k] <= self.upper[k] for k in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf:
    """
    Represents a terminal leaf node in the decision tree.

    Attributes:
        value (int): The predicted class or value at the leaf.
        depth (int): The depth of the leaf in the tree.
        is_leaf (bool): Always True for Leaf instances.
    """
    def __init__(self, value, depth):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Returns itself in a list, since it is already a leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaf has no children to propagate bounds to, so this is a no-op.
        """
        pass

    def update_indicator(self):
        """
        Leaf does not define its own indicator logic — this is set during propagation from a Node.
        """
        def is_large_enough(x):
            return np.all(np.array([x[:, k] > self.lower[k] for k in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, k] <= self.upper[k] for k in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Decision_Tree:
    """
    Wrapper class representing a full decision tree.

    Attributes:
        root (Node): The root node of the tree.
    """
    def __init__(self, root):
        self.root = root

    def get_leaves(self):
        """
        Returns all leaf nodes in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initiates recursive bounds computation from the root node.
        """
        self.root.update_bounds_below()