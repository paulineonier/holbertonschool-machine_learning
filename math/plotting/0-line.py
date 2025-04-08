#!/usr/bin/env python3
"""
This module defines the function line, which plots y = x³ as a solid red line.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a graph of y = x³ with the x-axis ranging from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(range(0, 11), y, 'r-')  # 'r-' specifies a red solid line
    plt.xlabel('x-axis')            # Label for the x-axis
    plt.ylabel('y-axis')            # Label for the y-axis
    plt.title('Graph of y = x³')    # Title of the graph
    plt.grid(True)                  # Add grid lines for clarity
    plt.show()                      # Display the graph
