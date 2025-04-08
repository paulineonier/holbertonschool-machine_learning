#!/usr/bin/env python3
"""
This module contains a function that plots the graph of y = x³.
The graph is plotted as a solid red line, and the x-axis ranges from 0 to 10.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    This function plots the graph of the function y = x³.
    It plots the curve as a solid red line, and the x-axis ranges from 0 to 10.
    """

    y = np.arange(0, 11) ** 3  # Compute y values for x from 0 to 10
    plt.figure(figsize=(6.4, 4.8))  # Set the size of the figure

    x = np.arange(0, 11)  # Define x values
    plt.plot(x, y, 'r-')  # Plot the curve y = x³ with a red solid line
    plt.xlim(0, 10)       # Set the x-axis limits from 0 to 10
    plt.show()            # Display the graph
