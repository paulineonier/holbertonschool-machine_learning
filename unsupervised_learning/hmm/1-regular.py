#!/usr/bin/env python3
"""Module for determining steady state of a regular Markov chain."""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): Square transition matrix of shape (n, n)

    Returns:
        numpy.ndarray: Steady state probabilities (1, n), or None on failure
    """
    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None

    n, m = P.shape
    if n != m:
        return None

    # Check probabilities
    if not np.allclose(P.sum(axis=1), 1):
        return None
    if np.any(P < 0):
        return None

    # Try to find steady state
    Pk = np.copy(P)

    for _ in range(1000):
        Pk_next = np.matmul(Pk, P)

        # Check convergence
        if np.allclose(Pk, Pk_next):
            # Check if regular (all positive)
            if np.all(Pk_next > 0):
                return Pk_next[0:1]
            return None

        Pk = Pk_next

    return None
