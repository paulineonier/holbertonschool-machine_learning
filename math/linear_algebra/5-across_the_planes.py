#!/usr/bin/env python3
"""
Module for element-wise addition of 2D matrices.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists of int/float): The first matrix to add.
        mat2 (list of lists of int/float): The second matrix to add.

    Returns:
        list of lists of int/float: New matrix representing element-wise sum
        of mat1 and mat2, or None if the matrices are not the same shape.
    """
    if len(mat1) != len(mat2):
        return None

    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    # Element-wise addition with updated variable name
    matrix_result = []
    for row1, row2 in zip(mat1, mat2):
        new_row = [elem1 + elem2 for elem1, elem2 in zip(row1, row2)]
        matrix_result.append(new_row)

    return matrix_result
