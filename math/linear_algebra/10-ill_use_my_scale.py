#!/usr/bin/env python3
"""
This module provides a function to calculate the shape of a numpy.ndarray.
"""

import numpy as np


def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): Input array whose shape is to be determined.

    Returns:
        tuple: A tuple of integers representing the dimensions of the array.
               For example:
               - A 1D array returns (size,).
               - A 2D array returns (rows, columns).
               - A 3D array returns (depth, rows, columns), etc.
    """
    return matrix.shape
