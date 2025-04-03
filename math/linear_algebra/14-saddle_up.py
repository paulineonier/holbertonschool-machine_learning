#!/usr/bin/env python3
"""
This module provides a function to perform matrix multiplication
on two numpy.ndarrays.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication on two numpy.ndarrays.

    Args:
        mat1 (numpy.ndarray): The first input matrix.
        mat2 (numpy.ndarray): The second input matrix.

    Returns:
        numpy.ndarray: Resulting matrix from the multiplication.
    """
    return np.matmul(mat1, mat2)
