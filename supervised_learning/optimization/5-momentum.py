#!/usr/bin/env python3
"""
Module that updates a variable using gradient descent with momentum.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum algorithm.

    Parameters:
    - alpha: float, the learning rate
    - beta1: float, the momentum weight (typically close to 1, e.g. 0.9)
    - var: numpy.ndarray, the variable to update
    - grad: numpy.ndarray, the gradient of var
    - v: numpy.ndarray, the previous moment (velocity)

    Returns:
    - The updated variable (numpy.ndarray)
    - The new moment (numpy.ndarray)
    """
    v_new = beta1 * v + (1 - beta1) * grad  # Update the velocity
    var_new = var - alpha * v_new           # Update variable using velocity

    return var_new, v_new
