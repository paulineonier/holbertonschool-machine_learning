#!/usr/bin/env python3

"""
Module that defines a function to calculate the cofactor matrix
of a given square matrix.
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a given square matrix.

    Args:
        matrix (list of lists): The matrix whose cofactor matrix is calculated.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The cofactor matrix of the input matrix.
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

    return cofactor_matrix
