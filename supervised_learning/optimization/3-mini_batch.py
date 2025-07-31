#!/usr/bin/env python3
"""creates mini-batches to train neural network"""

import numpy as np


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to train neural network using mini
    Batch gradient descent 

    Parameters:
    - X: numpy.ndarray of shape (m, nx), input data
    - Y: numpy.ndarray of shape (m, ny), representing the labels
        - m is the same number of data points as in X
        - ny is the number of classes for classification tasks
    - batch_size: the number of data points in a batch

    Returns:
    - Tuples (X_batch, Y_batch), list of mini-batches containing tuples
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # Create batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
