#!/usr/bin/env python3
"""
Module defines the function change_scale, which plots the exponential decay
of C-14 with a logarithmic scale on the y-axis.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14 with a logarithmically scaled y-axis.

    X-axis labeled 'Time (years)' and y-axis labeled 'Fraction Remaining'.
    The graph display the exponential decay formula for C-14 over time, with
    y-axis on a logarithmic scale. The x-axis will range from 0 to 28650 years.
    """
    x = np.arange(0, 28651, 5730)  # Time points in years, from 0 to 28650
    r = np.log(0.5)               # Decay constant (logarithm of 0.5)
    t = 5730                      # Half-life of C-14 in years
    y = np.exp((r / t) * x)       # Exponential decay formula

    plt.figure(figsize=(6.4, 4.8))  # Set the figure size
    plt.plot(x, y, 'b-')           # Plot the decay as a blue solid line
    plt.xlabel('Time (years)')     # Label for the x-axis
    plt.ylabel('Fraction Remaining')  # Label for the y-axis
    plt.title('Exponential Decay of C-14')  # Title of the graph
    plt.yscale('log')              # Set the y-axis to a logarithmic scale
    plt.xlim(0, 28650)             # Ensure x-axis ranges from 0 to 28650
    plt.grid(True)                 # Add grid lines for better readability
    plt.show()                     # Display the graph


change_scale()
