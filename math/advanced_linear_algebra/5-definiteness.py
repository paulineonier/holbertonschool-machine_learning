#!/usr/bin/env python3
"""
Module that defines a function to calculate the definiteness
of a given matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): Matrix whose definiteness is calculated.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.

    Returns:
        str or None: The definiteness of the matrix, or None if invalid.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (
        matrix.ndim != 2 or
        matrix.shape[0] != matrix.shape[1] or
        matrix.size == 0
    ):
        return None

    # Definiteness is defined only for symmetric matrices
    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
