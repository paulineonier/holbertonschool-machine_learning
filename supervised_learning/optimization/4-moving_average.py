#!/usr/bin/env python3
"""Function that calculates the weighted moving average of a data set"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Parameters:
    - data: list of data to calculate the moving average
    - beta: weight used for the moving average

    Returns:
    - list containing the moving average of data
    """
    v = 0
    moving_averages = []

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)

    return moving_averages
