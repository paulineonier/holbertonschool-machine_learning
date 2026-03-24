#!/usr/bin/env python3
"""
Module for performing K-means clustering
"""

import numpy as np


def initialize(X, k):
    """Initialisation des centroïdes (1er appel à uniform)"""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering

    Parameters:
    X (numpy.ndarray): shape (n, d)
    k (int): number of clusters
    iterations (int): max iterations

    Returns:
    C, clss or None, None on failure
    """
    # Validation
    if (not isinstance(X, np.ndarray) or
        len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    try:
        n, d = X.shape

        # Initialisation (1er appel uniform)
        C = initialize(X, k)

        # Pré-calcul min/max pour réinitialisation
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

        for _ in range(iterations):

            # Assignment step
            # distances shape: (n, k)
            distances = np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=2))
            clss = np.argmin(distances, axis=1)

            new_C = np.copy(C)

            # Update step
            for i in range(k):  # 2ème boucle
                points = X[clss == i]

                if len(points) == 0:
                    # Réinitialisation (2ème et dernier appel uniform)
                    new_C[i] = np.random.uniform(low=min_vals, high=max_vals)
                else:
                    new_C[i] = np.mean(points, axis=0)

            # Convergence check
            if np.allclose(C, new_C):
                return new_C, clss

            C = new_C

        return C, clss

    except Exception:
        return None, None
