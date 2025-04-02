#!/usr/bin/env python3
"""
Module: matrix_concatenation
Module provides function to concatenate two 2D matrices along specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of list): The first 2D matrix containing integers or floats.
        mat2 (list of list): Second 2D matrix containing integers or floats.
        axis (int): The axis along which to concatenate the matrices.

    Returns:
        list of list: A new matrix resulting from concatenating mat1 and mat2.
        None: If the matrices cannot be concatenated along the specified axis.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
