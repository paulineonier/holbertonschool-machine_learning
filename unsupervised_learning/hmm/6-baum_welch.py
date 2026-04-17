#!/usr/bin/env python3
"""Baum-Welch Algorithm Module.

This module implements the Baum-Welch algorithm (Expectation-Maximization)
for estimating the parameters of a Hidden Markov Model (HMM).
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Forward algorithm."""
    T = Observation.shape[0]
    N = Transition.shape[0]

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = np.sum(
            F[:, t - 1][:, np.newaxis] * Transition,
            axis=0
        ) * Emission[:, Observation[t]]

    return F


def backward(Observation, Emission, Transition):
    """Backward algorithm."""
    T = Observation.shape[0]
    N = Transition.shape[0]

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition *
            Emission[:, Observation[t + 1]] *
            B[:, t + 1],
            axis=1
        )

    return B


def baum_welch(Observations, Transition, Emission,
               Initial, iterations=1000):
    """Performs the Baum-Welch algorithm.

    Args:
        Observations (numpy.ndarray): Shape (T,)
        Transition (numpy.ndarray): Shape (M, M)
        Emission (numpy.ndarray): Shape (M, N)
        Initial (numpy.ndarray): Shape (M, 1)
        iterations (int): Number of EM iterations

    Returns:
        Transition (numpy.ndarray): Updated transition matrix
        Emission (numpy.ndarray): Updated emission matrix
        None, None: On failure
    """
    # Input validation
    if (not isinstance(Observations, np.ndarray) or
            not isinstance(Transition, np.ndarray) or
            not isinstance(Emission, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    if Observations.ndim != 1:
        return None, None

    if Transition.ndim != 2 or Emission.ndim != 2:
        return None, None

    if Initial.ndim != 2 or Initial.shape[1] != 1:
        return None, None

    M = Transition.shape[0]
    N = Emission.shape[1]
    T = Observations.shape[0]

    if Transition.shape != (M, M):
        return None, None

    if Emission.shape[0] != M:
        return None, None

    if Initial.shape[0] != M:
        return None, None

    for _ in range(iterations):
        # E-step
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition)

        P = np.sum(F[:, -1])
        if P == 0:
            return None, None

        gamma = (F * B) / P

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denom = np.sum(
                F[:, t][:, np.newaxis] *
                Transition *
                Emission[:, Observations[t + 1]] *
                B[:, t + 1]
            )

            if denom == 0:
                return None, None

            xi[:, :, t] = (
                F[:, t][:, np.newaxis] *
                Transition *
                Emission[:, Observations[t + 1]] *
                B[:, t + 1]
            ) / denom

        # M-step: update Transition
        Transition = np.sum(xi, axis=2) / np.sum(
            gamma[:, :-1], axis=1
        )[:, np.newaxis]

        # M-step: update Emission
        for k in range(N):
            mask = (Observations == k)
            Emission[:, k] = np.sum(
                gamma[:, mask], axis=1
            )

        Emission = Emission / np.sum(
            gamma, axis=1
        )[:, np.newaxis]

    return Transition, Emission
