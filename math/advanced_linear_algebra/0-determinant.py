#!/usr/bin/env python3
"""
Module 0-determinant

This module provides a function to calculate the determinant
of a square matrix.
The function supports matrices of any size, including the 0x0 matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list of lists of numbers): The square matrix
                        whose determinant is to be calculated.
            - The list [[]] represents a 0x0 matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.

    Returns:
        float or int: The determinant of the matrix.
    """
    # Vérification que matrix est une liste de listes
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Cas particulier : 0x0 matrix
    if matrix == [[]]:
        return 1

    n = len(matrix)

    # Vérifier que la matrice n'est pas vide
    if n == 0:
        raise TypeError("matrix must be a list of lists")

    # Vérification que la matrice est carrée
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Cas 1x1
    if n == 1:
        return matrix[0][0]

    # Cas 2x2
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Calcul récursif pour n > 2
    det = 0
    for c in range(n):
        # sous-matrice en supprimant la première ligne et la colonne c
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(submatrix)
    return det
