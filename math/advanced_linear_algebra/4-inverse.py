#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module that defines a function to calculate the inverse
of a given square matrix.
"""


def inverse(matrix):
    """
    Calculates the inverse of a given square matrix.

    Args:
        matrix (list of lists): The matrix whose inverse is calculated.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists or None: The inverse of the matrix, or None if singular.
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)

    if size == 1:
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

    def determinant(mat):
        """
        Computes the determinant of a square matrix.
        """
        n = len(mat)

        if n == 1:
            return mat[0][0]

        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0
        for col in range(n):
            submatrix = [
                row[:col] + row[col + 1:]
                for row in mat[1:]
            ]
            det += ((-1) ** col) * mat[0][col] * determinant(submatrix)

        return det

    det = determinant(matrix)

    if det == 0:
        return None

    # Compute cofactor matrix
    cofactor_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            submatrix = [
                matrix[r][:j] + matrix[r][j + 1:]
                for r in range(size) if r != i
            ]
            minor = determinant(submatrix)
            row.append(((-1) ** (i + j)) * minor)
        cofactor_matrix.append(row)

    # Transpose cofactor matrix (adjugate)
    adjugate_matrix = []
    for j in range(size):
        adj_row = []
        for i in range(size):
            adj_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(adj_row)

    # Divide by determinant
    inverse_matrix = []
    for row in adjugate_matrix:
        inverse_matrix.append([value / det for value in row])

    return inverse_matrix
