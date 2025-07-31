#!/usr/bin/env python3
"""Function that shuffles the data points of two matrices"""

import numpy as np

def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    - X: numpy.ndarray of shape (m, nx), the first matrix
    - Y: numpy.ndarray of shape (m, ny), the second matrix
        - m is the number of data points (rows)
        - nx is the number of features in X
        - ny is the number of features in Y

    Returns:
    - Tuple of (X_shuffled, Y_shuffled), both numpy.ndarrays of the same shape as input
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
