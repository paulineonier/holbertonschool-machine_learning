#!/usr/bin/env python3
import numpy as np

class Node:
    """
    A class representing a decision node in a binary decision tree.

    Attributes:
        feature (int): Index of the feature to split on.
        threshold (float): Threshold value for the split.
        left_child (Node or Leaf): Left child node or leaf.
        right_child (Node or Leaf): Right child node or leaf.
        depth (int): Depth of this node in the tree.
        is_root (bool): Whether this node is the root of the tree.
        is_leaf (bool): Always False for Node.
        lower (dict): Lower bounds for each feature (set during bounds computation).
        upper (dict): Upper bounds for each feature (set during bounds computation).
        indicator (function): A lambda function returning a boolean mask over data rows.
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
        Recursively collects all leaves in the subtree rooted at this node.

        Returns:
            list: A list of Leaf instances.
        """
        leaves = []
        for child in [self.left_child, self.right_child]:
            leaves.extend(child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively updates lower and upper bounds for each node or leaf below this node.
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
        Creates a lambda function (self.indicator) that returns a boolean array indicating
        whether each row of input data satisfies the bounds associated with this node.
        """
        def is_large_enough(x):
            return np.all(np.array([x[:, k] > self.lower[k] for k in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, k] <= self.upper[k] for k in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf:
    """
    A class representing a leaf node in a binary decision tree.

    Attributes:
        value (int or float): The prediction value associated with this leaf.
        depth (int): The depth of the leaf in the tree.
        is_leaf (bool): Always True for Leaf.
        lower (dict): Lower bounds for each feature (set during bounds computation).
        upper (dict): Upper bounds for each feature (set during bounds computation).
        indicator (function): A lambda function returning a boolean mask over data rows.
    """
    def __init__(self, value, depth):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Returns itself in a list since it is a leaf.

        Returns:
            list: A list containing only this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Does nothing for leaves as they have no children.
        """
        pass

    def update_indicator(self):
        """
        Creates a lambda function (self.indicator) that returns a boolean array indicating
        whether each row of input data satisfies the bounds associated with this leaf.
        """
        def is_large_enough(x):
            return np.all(np.array([x[:, k] > self.lower[k] for k in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, k] <= self.upper[k] for k in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Decision_Tree:
    """
    A class representing a full binary decision tree.

    Attributes:
        root (Node): The root node of the tree.
    """
    def __init__(self, root):
        self.root = root

    def get_leaves(self):
        """
        Returns all leaves of the tree.

        Returns:
            list: A list of all Leaf instances in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initiates recursive lower and upper bounds update from the root.
        """
        self.root.update_bounds_below()