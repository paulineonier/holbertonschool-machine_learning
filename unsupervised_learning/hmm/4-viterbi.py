#!/usr/bin/env python3
"""Viterbi Algorithm Module.

This module provides a function to compute the most likely sequence
of hidden states for a Hidden Markov Model (HMM) using the Viterbi
dynamic programming algorithm.
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Performs the Viterbi algorithm.

    Computes the most likely sequence of hidden states given an
    observation sequence and the parameters of a Hidden Markov Model.

    Args:
        Observation (numpy.ndarray): Shape (T,), indices of observations.
        Emission (numpy.ndarray): Shape (N, M), emission probabilities.
        Transition (numpy.ndarray): Shape (N, N), transition probabilities.
        Initial (numpy.ndarray): Shape (N, 1), initial state probabilities.

    Returns:
        path (list): Most likely sequence of hidden states (length T).
        P (float): Probability of the optimal path.
        None, None: On failure.
    """
    # Input validation
    if (not isinstance(Observation, np.ndarray) or
            not isinstance(Emission, np.ndarray) or
            not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    if Observation.ndim != 1:
        return None, None

    if Emission.ndim != 2 or Transition.ndim != 2:
        return None, None

    if Initial.ndim != 2 or Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N):
        return None, None

    if Initial.shape[0] != N:
        return None, None

    # Initialize DP tables
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    # Initialization step
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Recursion step
    for t in range(1, T):
        for j in range(N):
            prob = V[:, t - 1] * Transition[:, j]
            B[j, t] = np.argmax(prob)
            V[j, t] = np.max(prob) * Emission[j, Observation[t]]

    # Termination step
    last_state = np.argmax(V[:, T - 1])
    P = V[last_state, T - 1]

    # Path backtracking
    path = [0] * T
    path[T - 1] = last_state

    for t in range(T - 2, -1, -1):
        path[t] = B[path[t + 1], t + 1]

    return path, P
