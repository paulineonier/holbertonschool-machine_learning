#!/usr/bin/env python3
"""
Module that calculates the posterior probability of each
hypothetical probability of developing severe side effects
given observed data and prior beliefs.
"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for each hypothetical
    probability given the observed data.

    Parameters
    ----------
    x : int
        Number of patients that develop severe side effects
    n : int
        Total number of patients observed
    P : numpy.ndarray
        1D array containing hypothetical probabilities
    Pr : numpy.ndarray
        1D array containing prior beliefs of P

    Returns
    -------
    numpy.ndarray
        Posterior probability of each probability in P
    """

    # ----- INPUT VALIDATION -----
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    #  CALCUL DU COEFFICIENT BINOMIAL
    factorial = np.math.factorial
    coeff = factorial(n) / (
        factorial(x) * factorial(n - x)
    )

    #  LIKELIHOOD
    likelihoods = coeff * (P ** x) * ((1 - P) ** (n - x))

    #  INTERSECTION
    intersection = likelihoods * Pr

    #  MARGINAL
    marginal_prob = np.sum(intersection)

    #  POSTERIOR
    posterior_prob = intersection / marginal_prob

    return posterior_prob
