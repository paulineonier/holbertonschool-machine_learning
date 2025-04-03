#!/usr/bin/env python3
"""
This module provides a function to calculate the shape of numpy.ndarray.
"""


def np_shape(matrix):
    """
    Calculates the shape of numpy.ndarray.

    Args:
        matrix (numpy.ndarray): Input array, shape is to be determined.

    Returns:
        tuple: Tuple of integers representing dimensions of numpy.ndarray.
               For example:
               - A 1D list returns (size,).
               - A 2D list returns (rows, columns).
               - A 3D list returns (depth, rows, columns), etc.
    """
    return matrix.shape
