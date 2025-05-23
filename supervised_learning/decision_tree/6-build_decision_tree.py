#!/usr/bin/env python3
import numpy as np

class Node:
    """
    A decision node in a binary decision tree.

    Attributes:
        feature (int): The feature index this node splits on.
        threshold (float): The threshold value for the split.
        left_child (Node or Leaf): Left child node (if feature <= threshold).
        right_child (Node or Leaf): Right child node (if feature > threshold).
        depth (int): Depth of the node in the tree.
        is_root (bool): Flag indicating if the node is the root.
        is_leaf (bool): Always False for Node.
        lower (dict): Lower bounds on feature values (set during bounds computation).
        upper (dict): Upper bounds on feature values (set during bounds computation).
        indicator (callable): Function that returns a boolean array indicating if each sample matches the node's region.
    """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def __str__(self):
        return f"{'root' if self.is_root else 'node'} [feature={self.feature}, threshold={self.threshold}]"

    def get_leaves_below(self):
        """
        Recursively collect all leaves in the subtree rooted at this node.
        """
        return self.left_child.get_leaves_below() + self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """
        Propagate lower and upper feature bounds to child nodes or leaves.
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

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Define an indicator function for the node based on feature bounds.
        """
        def is_large_enough(x):
            return np.all(np.array([x[:, k] > self.lower[k] for k in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, k] <= self.upper[k] for k in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Recursively predict the label for a single observation x using the tree structure.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf:
    """
    A terminal leaf node in a binary decision tree.

    Attributes:
        value (int or float): The predicted value of this leaf.
        depth (int): Depth in the tree.
        is_leaf (bool): Always True.
        lower (dict): Lower feature bounds (set during tree bounds computation).
        upper (dict): Upper feature bounds (set during tree bounds computation).
        indicator (callable): Boolean function checking which samples fall into this leaf.
    """
    def __init__(self, value, depth):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def update_indicator(self):
        def is_large_enough(x):
            return np.all(np.array([x[:, k] > self.lower[k] for k in self.lower]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([x[:, k] <= self.upper[k] for k in self.upper]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Return the value of this leaf for a single observation.
        """
        return self.value


class Decision_Tree:
    """
    A binary decision tree structure composed of Nodes and Leaves.

    Attributes:
        root (Node): Root node of the tree.
        predict (callable): Efficient vectorized prediction function defined via update_predict.
    """
    def __init__(self, root):
        self.root = root

    def get_leaves(self):
        """
        Return all leaves of the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update bounds across all nodes and leaves of the tree.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Defines a vectorized prediction function by first updating all bounds and indicators,
        and then assigning the correct value from the corresponding leaf.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value for leaf in leaves for i in range(A.shape[0]) if leaf.indicator(A)[i]])

    def pred(self, x):
        """
        Predict a label for a single observation using recursive tree traversal.
        """
        return self.root.pred(x)