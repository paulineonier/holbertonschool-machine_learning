#!/usr/bin/env python3
"""
Module for initializing cluster centroids for K-means clustering
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    X (numpy.ndarray): shape (n, d)
    k (int): number of clusters

    Returns:
    numpy.ndarray of shape (k, d) containing initialized centroids,
    or None on failure
    """
    # Validate inputs
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or not isinstance(k, int) or k <= 0):
        return None

    try:
        _, d = X.shape

        # Compute min and max per dimension
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

        # Initialize centroids
        centroids = np.random.uniform(
            low=min_vals,
            high=max_vals,
            size=(k, d)
        )

        return centroids

    except Exception:
        return None
