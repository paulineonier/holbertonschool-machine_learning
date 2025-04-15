#!/usr/bin/env python3

"""
Module: polynomial_utils
This module provides utility functions for working with polynomials,
including differentiation.
"""


def poly_derivative(poly):
    """
    Calcule dérivée d'un polynôme représenté par liste de coefficients.

    Args:
        poly (list): Liste de coefficients représentant un polynôme.
                     L'indice = puissance de x du coefficient.

    Returns:
        list: Liste de coefficients représentant la dérivée du polynôme.
              Si la dérivée est nulle, retourne [0].
        None: Si poly = pas une liste valide

    Exemple:
        Si f(x) = x^3 + 3x + 5, alors poly = [5, 3, 0, 1]
        La fonction retourne [3, 0, 3], car f'(x) = 3x^2 + 0x + 3.

    Notes:
        - Si poly est vide ou ne contient que des constantes,
          la dérivée est [0].
    """
    if not isinstance(poly, list) or not poly:
        return None

    derivative = [i * coef for i, coef in enumerate(poly) if i > 0]

    return derivative if derivative else [0]
