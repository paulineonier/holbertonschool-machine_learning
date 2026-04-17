#!/usr/bin/env python3
"""Backward Algorithm Module.

This module implements the backward algorithm for a Hidden Markov Model
(HMM). It computes the likelihood of an observation sequence given the
model and the backward probabilities matrix.
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): Shape (T,), indices of observations.
        Emission (numpy.ndarray): Shape (N, M), emission probabilities.
        Transition (numpy.ndarray): Shape (N, N), transition probabilities.
        Initial (numpy.ndarray): Shape (N, 1), initial state probabilities.

    Returns:
        P (float): Likelihood of the observations given the model.
        B (numpy.ndarray): Shape (N, T), backward probabilities.
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

    # Initialize backward matrix
    B = np.zeros((N, T))

    # Initialization step
    B[:, T - 1] = 1

    # Recursion step
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                Transition[i, :] *
                Emission[:, Observation[t + 1]] *
                B[:, t + 1]
            )

    # Termination step
    P = np.sum(
        Initial[:, 0] *
        Emission[:, Observation[0]] *
        B[:, 0]
    )

    return P, B
