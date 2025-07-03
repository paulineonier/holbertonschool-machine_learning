#!/usr/bin/env python3
"""
Function to convert a one-hot encoded matrix back to numeric labels.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of numeric labels.

    Parameters
    ----------
    one_hot : numpy.ndarray of shape (classes, m)
        One-hot encoded matrix.

    Returns
    -------
    numpy.ndarray of shape (m,)
        Numeric labels for each example.
    None
        On failure.
    """
    try:
        # The label is the index of the max value in each column
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
