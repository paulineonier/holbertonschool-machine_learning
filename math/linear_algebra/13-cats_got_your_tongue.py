#!/usr/bin/env python3
"""
This module provides a function to concatenate two numpy.ndarrays
along a specific axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy.ndarrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): The first input matrix.
        mat2 (numpy.ndarray): The second input matrix.
        axis (int, optional): Axis along which to concatenate. Default 0.

    Returns:
        numpy.ndarray: A new array obtained by concatenating mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
