#!/usr/bin/env python3
"""
This module provides a function to perform element-wise addition,
subtraction, multiplication, and division on numpy.ndarrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, division
    on two numpy.ndarrays.

    Args:
        mat1 (numpy.ndarray): The first input array.
        mat2 (numpy.ndarray or scalar): The second input array or scalar.

    Returns:
        tuple: A tuple containing the element-wise sum, difference,
               product, and quotient of mat1 and mat2.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
