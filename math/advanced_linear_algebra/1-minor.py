#!/usr/bin/env python3

"""
Module that defines a function to calculate the minor matrix
of a given square matrix.
"""


def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): The matrix whose minor matrix is calculated.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The minor matrix of the input matrix.
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)

    if size == 1:
        return [[1]]

    def determinant(mat):
        """
        Computes the determinant of a square matrix.

        Args:
            mat (list of lists): Square matrix.

        Returns:
            int or float: Determinant of the matrix.
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

    minor_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            submatrix = [
                matrix[r][:j] + matrix[r][j + 1:]
                for r in range(size) if r != i
            ]
            row.append(determinant(submatrix))
        minor_matrix.append(row)

    return minor_matrix
