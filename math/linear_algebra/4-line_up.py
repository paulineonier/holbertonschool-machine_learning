#!/usr/bin/env python3

"""
This module provides basic linear algebra operations for arrays.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): A list of integers or floats.
        arr2 (list): A list of integers or floats.

    Returns:
        list: A new list containing the element-wise sum of arr1 and arr2,
        or None if their shapes do not match.
    """
    if len(arr1) != len(arr2):
        return None

    return [arr1[i] + arr2[i] for i in range(len(arr1))]
