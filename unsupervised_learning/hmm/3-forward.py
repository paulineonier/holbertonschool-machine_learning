#!/usr/bin/env python3
"""Module to determine if a Markov chain is absorbing."""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P (numpy.ndarray): Transition matrix (n, n)

    Returns:
        bool: True if absorbing, False otherwise
    """
    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False

    n, m = P.shape
    if n != m:
        return False

    # Check probabilities
    if not np.allclose(P.sum(axis=1), 1):
        return False
    if np.any(P < 0):
        return False

    # Find absorbing states
    absorbing_states = np.isclose(np.diag(P), 1)

    if not np.any(absorbing_states):
        return False

    # Reachability check
    reachable = np.copy(P)

    # Compute reachability via powers
    for _ in range(n):
        reachable = np.matmul(reachable, P)

    # Check if every state can reach an absorbing state
    for i in range(n):
        if not np.any(reachable[i][absorbing_states] > 0):
            return False

    return True
