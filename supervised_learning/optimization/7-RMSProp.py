#!/usr/bin/env python3
"""
RMSProp optimizer update function
"""

import numpy as np

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    Parameters:
    - alpha (float): learning rate
    - beta2 (float): decay rate for the moving average of squared gradients
    - epsilon (float): small constant to avoid division by zero
    - var (np.ndarray or float): the variable to update (weights or bias)
    - grad (np.ndarray or float): gradient of the cost with respect to var
    - s (np.ndarray or float): the previous second moment of the gradients

    Returns:
    - var (np.ndarray or float): updated variable
    - s (np.ndarray or float): updated second moment
    """
    # Update the exponentially weighted moving average of the squared gradient
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Apply RMSProp update rule
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
