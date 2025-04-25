#!/usr/bin/env python3
"""
Module: Polynomial Integration

Module provides a function `poly_integral` that computes integral of polynomial
represented as a list of coefficients. Index of list corresponds to power of x
associated with each coefficient.
"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing the polynomial.
                     The index of the list represents the power of x.
                     Example: [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.
        C (int or float, optional): The integration constant. Defaults to 0.

    Returns:
        list: New list of coefficients representing the integral of polynomial.
              If a coefficient is an integer, it is represented as an int.
        None: If `poly` or `C` are invalid.
    """
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        if coeff.is_integer():
            coeff = int(coeff)
        integral.append(coeff)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral