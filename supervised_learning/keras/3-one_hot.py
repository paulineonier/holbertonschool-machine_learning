#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix"""

import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels (np.ndarray): Array of class labels (integers)
        classes (int): Total number of classes. If None, use max label + 1.

    Returns:
        np.ndarray: One-hot encoded matrix
    """
    if classes is None:
        classes = np.max(labels) + 1

    return np.eye(classes)[labels]
