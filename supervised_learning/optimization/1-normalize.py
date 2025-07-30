#!/usr/bin/env python3
"""Module that normalizes (standardizes) a matrix"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    - X: numpy.ndarray of shape (d, nx) to normalize
    - m: numpy.ndarray of shape (nx,) containing the mean of each feature
    - s: numpy.ndarray of shape (nx,) containing the standard deviation of each feature
s
    Returns:
    - The normalized X matrix
    """
    return (X - m) / s
