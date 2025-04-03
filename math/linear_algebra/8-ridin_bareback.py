#!/usr/bin/env python3
"""
Module: matrix_multiplication
Provides a function to perform matrix multiplication for 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices if possible.

    Args:
        mat1 (list of list): The first 2D matrix containing integers or floats.
        mat2 (list of list): Second 2D matrix containing integers or floats.

    Returns:
        list of list: A new matrix resulting from multiplying mat1 and mat2.
        None: If the matrices cannot be multiplied.
    """
    # Check if multiplication is possible: columns in mat1 == rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the resulting matrix with zero values
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):  # Iterate through rows of mat1
        for j in range(len(mat2[0])):  # Iterate through columns of mat2
            for k in range(len(mat2)):  # Iterate through rows of mat2
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
