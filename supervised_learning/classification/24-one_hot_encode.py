#!/usr/bin/env python3
"""
Function to convert numeric label vector into one-hot encoded matrix.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters
    ----------
    Y : numpy.ndarray of shape (m,)
        Numeric class labels.
    classes : int
        Maximum number of classes found in Y.

    Returns
    -------
    numpy.ndarray of shape (classes, m)
        One-hot encoding of Y.
    None
        On failure.
    """
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
