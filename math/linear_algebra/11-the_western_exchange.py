#!/usr/bin/env python3
"""
This module provides a function to transpose a numpy.ndarray.
"""


def np_transpose(matrix):
    """
    Transposes numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The input array to be transposed.

    Returns:
        numpy.ndarray: A new array that is the transpose of the input.
    """
    return matrix.T
