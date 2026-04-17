#!/usr/bin/env python3
"""Module for computing Markov chain state probabilities."""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a specific
    state after a given number of iterations.

    Args:
        P (numpy.ndarray): Square transition matrix of shape (n, n),
            where P[i, j] is the probability of transitioning from
            state i to state j.
        s (numpy.ndarray): Initial state distribution of shape (1, n).
        t (int): Number of iterations.

    Returns:
        numpy.ndarray: Probability distribution after t iterations,
            shape (1, n), or None on failure.
    """
    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n, m = P.shape
    if n != m:
        return None

    # Validate s
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape != (1, n):
        return None

    # Validate t
    if not isinstance(t, int) or t < 0:
        return None

    # Validate probabilities in P
    if not np.allclose(P.sum(axis=1), 1):
        return None
    if np.any(P < 0):
        return None

    # Validate probabilities in s
    if not np.isclose(s.sum(), 1):
        return None
    if np.any(s < 0):
        return None

    # Compute P^t
    P_t = np.linalg.matrix_power(P, t)

    # Compute final state
    return np.matmul(s, P_t)
